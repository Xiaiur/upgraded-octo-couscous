import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.tensorboard import SummaryWriter

from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel

# -------------------- Утилиты для логирования --------------------
class TrainingLogger:
    """Класс для логирования процесса обучения"""
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = None):
        self.log_dir = log_dir
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_path, exist_ok=True)
        
        # Инициализация TensorBoard
        self.tb_writer = SummaryWriter(log_dir=self.experiment_path)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.steps = 0
        self.epochs = 0
        
        print(f"[Logger] Логи будут сохранены в: {self.experiment_path}")
    
    def log_step(self, 
                 loss: float, 
                 learning_rate: float = None,
                 grad_norm: float = None,
                 step: int = None):
        """Логирование на каждом шаге"""
        if step is not None:
            self.steps = step
        else:
            self.steps += 1
            
        self.train_losses.append(loss)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
            self.tb_writer.add_scalar('train/learning_rate', learning_rate, self.steps)
        
        if grad_norm is not None:
            self.tb_writer.add_scalar('train/grad_norm', grad_norm, self.steps)
        
        self.tb_writer.add_scalar('train/loss', loss, self.steps)
        self.tb_writer.add_scalar('train/step', self.steps, self.steps)
    
    def log_epoch(self, 
                  epoch: int, 
                  train_loss: float = None,
                  val_loss: float = None,
                  metrics: Dict = None):
        """Логирование в конце эпохи"""
        self.epochs = epoch
        self.tb_writer.add_scalar('epoch', epoch, self.steps)
        
        if train_loss is not None:
            self.tb_writer.add_scalar('train/epoch_loss', train_loss, epoch)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            self.tb_writer.add_scalar('val/loss', val_loss, epoch)
        
        if metrics:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f'val/{key}', value, epoch)
        
        # Периодическое создание графиков
        if epoch % 1 == 0:  # Каждую эпоху
            self.plot_progress()
    
    def plot_progress(self):
        """Создание и сохранение графиков прогресса"""
        if not self.train_losses:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. График потерь обучения
        axes[0].plot(self.train_losses, alpha=0.7, linewidth=1)
        axes[0].set_xlabel('Шаги')
        axes[0].set_ylabel('Потери обучения')
        axes[0].set_title(f'Потери обучения (шаг {self.steps})')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Сглаженные потери (moving average)
        if len(self.train_losses) > 50:
            window = min(100, len(self.train_losses) // 10)
            moving_avg = pd.Series(self.train_losses).rolling(window=window).mean()
            axes[1].plot(moving_avg, label=f'Скользящее среднее (окно={window})', 
                        color='red', linewidth=2)
            axes[1].plot(self.train_losses, alpha=0.3, label='Сырые значения')
            axes[1].set_xlabel('Шаги')
            axes[1].set_ylabel('Потери')
            axes[1].set_title('Сглаженные потери обучения')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Валидационные потери (если есть)
        if self.val_losses:
            epochs = range(len(self.val_losses))
            axes[2].plot(epochs, self.val_losses, 'o-', label='Валидационные потери')
            axes[2].set_xlabel('Эпоха')
            axes[2].set_ylabel('Потери валидации')
            axes[2].set_title('Потери валидации по эпохам')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # 4. Learning rate (если есть)
        if self.learning_rates:
            steps = range(len(self.learning_rates))
            axes[3].plot(steps, self.learning_rates, color='green')
            axes[3].set_xlabel('Шаги')
            axes[3].set_ylabel('Learning Rate')
            axes[3].set_title('Динамика Learning Rate')
            axes[3].set_yscale('log')
            axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(f'Прогресс обучения (Эпоха {self.epochs}, Шаги {self.steps})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Сохранение в TensorBoard
        self.tb_writer.add_figure('training_progress', fig, self.steps)
        
        # Сохранение как изображение
        plot_path = os.path.join(self.experiment_path, f'training_progress_epoch_{self.epochs}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_final_summary(self, model_name: str, config: Dict):
        """Создание финального суммарного графика"""
        if not self.train_losses:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Полная история потерь
        axes[0].plot(self.train_losses, alpha=0.8, linewidth=1)
        if self.val_losses:
            val_steps = np.linspace(0, len(self.train_losses)-1, len(self.val_losses))
            axes[0].plot(val_steps, self.val_losses, 'o-', markersize=4, 
                        label='Валидация', linewidth=2)
        axes[0].set_xlabel('Шаги')
        axes[0].set_ylabel('Потери')
        axes[0].set_title('Полная история обучения')
        axes[0].grid(True, alpha=0.3)
        if self.val_losses:
            axes[0].legend()
        
        # 2. Сглаженные потери с окнами
        axes[1].plot(self.train_losses, alpha=0.3, label='Сырые значения')
        windows = [10, 50, 100]
        colors = ['red', 'blue', 'green']
        for window, color in zip(windows, colors):
            if len(self.train_losses) > window:
                moving_avg = pd.Series(self.train_losses).rolling(window=window).mean()
                axes[1].plot(moving_avg, label=f'Окно={window}', color=color, linewidth=2)
        axes[1].set_xlabel('Шаги')
        axes[1].set_ylabel('Потери')
        axes[1].set_title('Сглаженные потери с разными окнами')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Потери по эпохам (если есть валидация)
        if self.val_losses:
            epochs = range(1, len(self.val_losses) + 1)
            axes[2].bar(epochs, self.val_losses, alpha=0.7)
            axes[2].set_xlabel('Эпоха')
            axes[2].set_ylabel('Потери валидации')
            axes[2].set_title('Валидационные потери по эпохам')
            axes[2].grid(True, alpha=0.3, axis='y')
            # Добавление значений на столбцы
            for i, v in enumerate(self.val_losses):
                axes[2].text(i + 1, v, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=8)
        else:
            # Альтернативный график: распределение потерь
            axes[2].hist(self.train_losses, bins=50, alpha=0.7)
            axes[2].set_xlabel('Значения потерь')
            axes[2].set_ylabel('Частота')
            axes[2].set_title('Распределение значений потерь')
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Финальный отчет: {model_name}\nКонфигурация: {config}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Сохранение
        final_plot_path = os.path.join(self.experiment_path, 'final_training_summary.png')
        plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    def save_logs(self):
        """Сохранение логов в файл"""
        log_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'total_steps': self.steps,
            'total_epochs': self.epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        log_file = os.path.join(self.experiment_path, 'training_logs.json')
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Также сохраняем как CSV для удобства
        df = pd.DataFrame({
            'step': range(len(self.train_losses)),
            'train_loss': self.train_losses,
            'learning_rate': (self.learning_rates + 
                            [None] * (len(self.train_losses) - len(self.learning_rates))) 
                            if self.learning_rates else [None] * len(self.train_losses)
        })
        csv_file = os.path.join(self.experiment_path, 'training_logs.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"[Logger] Логи сохранены в: {log_file}")
        print(f"[Logger] CSV сохранен в: {csv_file}")
    
    def close(self):
        """Закрытие логгера"""
        self.tb_writer.close()
        self.save_logs()


class CustomTrainerCallback(TrainerCallback):
    """Кастомный callback для интеграции с Trainer"""
    
    def __init__(self, logger: TrainingLogger):
        self.logger = logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при логировании"""
        if logs is not None:
            loss = logs.get('loss')
            learning_rate = logs.get('learning_rate')
            grad_norm = logs.get('grad_norm')
            
            if loss is not None:
                self.logger.log_step(
                    loss=loss,
                    learning_rate=learning_rate,
                    grad_norm=grad_norm,
                    step=state.global_step
                )
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце эпохи"""
        epoch = state.epoch
        train_loss = state.log_history[-1].get('loss') if state.log_history else None
        
        self.logger.log_epoch(
            epoch=int(epoch),
            train_loss=train_loss
        )


# -------------------- Утилиты --------------------
def read_markdown(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def prepare_markdown_dataset(
    text: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    stride: int = 128,
    test_size: float = 0.1
):
    """Подготовка датасета из markdown с разделением на train/validation"""
    tokenized = tokenizer(
        text,
        return_tensors=None,
        add_special_tokens=False
    )["input_ids"]

    records = []

    for start in range(0, len(tokenized) - max_length, max_length - stride):
        chunk = tokenized[start:start + max_length]

        records.append({
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
            "labels": chunk.copy(),
        })

    # Разделяем на train и validation
    total_samples = len(records)
    split_idx = int(total_samples * (1 - test_size))
    
    train_records = records[:split_idx]
    val_records = records[split_idx:] if test_size > 0 else []
    
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records) if val_records else None
    
    print(f"Создано {len(train_records)} train и {len(val_records)} validation примеров")
    
    return train_ds, val_ds

def read_json_or_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if not text:
            return []
        # Попытка загрузить как JSON целиком (список)
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        # Иначе обрабатываем как JSONL
        items = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items


def get_default_lora_targets(model_name: str) -> List[str]:
    mn = model_name.lower()
    if 'llama' in mn or 'alpaca' in mn:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if 'gpt2' in mn or 'gpt' in mn or 'dialo' in mn:
        return ["c_attn", "c_proj", "c_fc", "c_ffn"]
    # Qwen/Falcon/иные: стандартная попытка
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


# -------------------- Подготовка данных --------------------
def prepare_dataset(items: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 512, test_size: float = 0.1):
    """
    Корректно токенизируем prompt и response отдельно, собираем input_ids и labels
    labels: -100 для токенов prompt (и sep), реальные id для токенов response
    Возвращает train и validation датасеты
    """
    records = []

    # выберем SEP как eos_token (если есть), иначе "\n"
    if tokenizer.eos_token_id is not None:
        sep_token_id = tokenizer.eos_token_id
    else:
        # как fallback — добавим перенос строки (будет токенизирован)
        sep_token_id = None

    for it in items:
        prompt = it.get('context', '') or ''
        response = it.get('utterance', '') or ''

        # токенизируем отдельно без добавления специальных токенов
        enc_prompt = tokenizer(prompt, add_special_tokens=False)
        enc_resp = tokenizer(response, add_special_tokens=False)

        prompt_ids = enc_prompt["input_ids"]
        resp_ids = enc_resp["input_ids"]

        # составляем input_ids: prompt + [sep?] + response
        if sep_token_id is not None:
            input_ids = prompt_ids + [sep_token_id] + resp_ids
            prompt_len = len(prompt_ids) + 1
        else:
            # если нет eos_token_id — положим явный разделитель как перевод строки и токенизируем
            sep_enc = tokenizer("\n", add_special_tokens=False)
            sep_ids = sep_enc["input_ids"]
            input_ids = prompt_ids + sep_ids + resp_ids
            prompt_len = len(prompt_ids) + len(sep_ids)

        # усекаем до max_length справа
        input_ids = input_ids[-max_length:] if len(input_ids) > max_length else input_ids

        # если усекли — надо пересчитать prompt_len корректно (в случае, когда усечена часть prompt)
        if prompt_len > len(input_ids):
            # весь prompt не поместился — тогда ничего размечать как trainable
            labels = [-100] * len(input_ids)
        else:
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            # выравнивание на случай рассинхронов
            if len(labels) < len(input_ids):
                labels = labels + [-100] * (len(input_ids) - len(labels))
            elif len(labels) > len(input_ids):
                labels = labels[:len(input_ids)]

        # attention_mask — все 1 (будет паддиться в collator)
        attention_mask = [1] * len(input_ids)

        records.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    # Разделяем на train и validation
    total_samples = len(records)
    split_idx = int(total_samples * (1 - test_size))
    
    train_records = records[:split_idx]
    val_records = records[split_idx:] if test_size > 0 else []
    
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records) if val_records else None
    
    print(f"Создано {len(train_records)} train и {len(val_records)} validation примеров")
    
    return train_ds, val_ds


@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # 1. Собираем списки
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # 2. Находим max длину внутри батча
        max_len = max(x.size(0) for x in input_ids)

        # 3. Паддим вручную
        def pad(tensor, pad_value):
            return torch.nn.functional.pad(
                tensor,
                (0, max_len - tensor.size(0)),
                value=pad_value
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # fallback
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids = torch.stack([pad(x, pad_id) for x in input_ids])
        attention_mask = torch.stack([pad(x, 0) for x in attention_mask])
        labels = torch.stack([pad(x, -100) for x in labels])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune causal LM with LoRA (PEFT)')

    parser.add_argument('--model_name', type=str, default='gpt2', help='Название модели в HF')
    parser.add_argument('--data_path', type=str, required=True, help='Путь к data.json или data.jsonl или .md файлу')
    parser.add_argument('--output_dir', type=str, default='./fine-tuned-lora', help='Куда сохранить модель/адаптер')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Директория для логов')
    parser.add_argument('--experiment_name', type=str, default=None, help='Имя эксперимента')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logging_steps', type=int, default=10, help='Частота логирования')
    parser.add_argument('--eval_steps', type=int, default=100, help='Частота валидации (0 - отключить валидацию)')
    parser.add_argument('--save_steps', type=int, default=500, help='Частота сохранения')
    parser.add_argument('--test_size', type=float, default=0.1, help='Доля данных для валидации (0.0-0.3)')
    parser.add_argument('--no_eval', action='store_true', help='Полностью отключить валидацию')

    args = parser.parse_args()
    
    # Инициализация логгера
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    # Сбор конфигурации для отчета
    config = {
        'model_name': args.model_name,
        'data_path': args.data_path,
        'use_lora': args.use_lora,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'learning_rate': args.learning_rate,
        'batch_size': args.per_device_train_batch_size,
        'num_epochs': args.num_train_epochs,
        'max_length': args.max_length,
        'seed': args.seed,
        'test_size': args.test_size,
        'eval_steps': args.eval_steps
    }
    
    print(f"Конфигурация обучения:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"Загрузка токенизатора и модели: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Гарантируем наличие pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Загружаем базовую модель
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # для тренировки выключаем кэш генерации
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    model.config.use_cache = False

    # Настройка LoRA
    if args.use_lora:
        target_modules = get_default_lora_targets(args.model_name)
        print(f"Используем target_modules для LoRA: {target_modules}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        # покажем сколько параметров обучается
        model.print_trainable_parameters()
    
    # Определяем тип данных и загружаем
    if args.data_path.endswith('.md'):
        train_ds, eval_ds = prepare_markdown_dataset(
            text=read_markdown(args.data_path),
            tokenizer=tokenizer,
            max_length=args.max_length,
            test_size=0 if args.no_eval else args.test_size
        )
    else:
        items = read_json_or_jsonl(args.data_path)
        train_ds, eval_ds = prepare_dataset(
            items, 
            tokenizer, 
            args.max_length,
            test_size=0 if args.no_eval else args.test_size
        )
    
    print(f"Размер тренировочного датасета: {len(train_ds)} примеров")
    if eval_ds:
        print(f"Размер валидационного датасета: {len(eval_ds)} примеров")
    else:
        print("Валидационный датасет не создан")

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    # Определяем стратегию валидации
    if args.no_eval or args.eval_steps <= 0 or eval_ds is None:
        eval_strategy = "no"
        eval_steps = None
        load_best_model_at_end = False
        print("Валидация отключена")
    else:
        eval_strategy = "steps"
        eval_steps = args.eval_steps
        load_best_model_at_end = True
        print(f"Валидация включена каждые {eval_steps} шагов")

    # Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy=eval_strategy,
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="loss" if eval_ds else None,
        greater_is_better=False,
        logging_dir=logger.experiment_path,
        seed=args.seed,
        gradient_accumulation_steps=1,
        warmup_steps=100,
    )

    # Создаем Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "data_collator": data_collator,
        "callbacks": [CustomTrainerCallback(logger)],
    }
    
    if eval_ds and eval_strategy != "no":
        trainer_kwargs["eval_dataset"] = eval_ds
    
    trainer = Trainer(**trainer_kwargs)

    print("Начинаем обучение...")
    
    try:
        train_result = trainer.train()
        
        # Сохраняем метрики обучения
        metrics = train_result.metrics
        metrics_file = os.path.join(logger.experiment_path, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Метрики обучения сохранены в: {metrics_file}")
        print(f"Финальные метрики: {metrics}")
        
    except Exception as e:
        print(f"Ошибка во время обучения: {e}")
        raise
    finally:
        # Всегда закрываем логгер
        logger.close()

    print("Сохраняем результат...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Если использовали LoRA: сохраняем ТОЛЬКО адаптер
    if args.use_lora:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Сохранён LoRA-адаптер и токенизатор в {args.output_dir}")
    else:
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Сохранена полная модель и токенизатор в {args.output_dir}")

    # Создаем финальный отчет
    logger.plot_final_summary(args.model_name, config)
    
    # Копируем логи в output_dir
    import shutil
    logs_in_output = os.path.join(args.output_dir, 'training_logs')
    shutil.copytree(logger.experiment_path, logs_in_output, dirs_exist_ok=True)
    print(f"Логи обучения также сохранены в: {logs_in_output}")

    print("Готово.")


if __name__ == '__main__':
    main()
