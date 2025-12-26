import sys
import warnings
warnings.filterwarnings('ignore')

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
        
        # Автоматическое определение среды выполнения
        self.is_colab = 'google.colab' in sys.modules
        if self.is_colab:
            # В Colab лучше логировать в /content
            self.log_dir = '/content/logs'
            print(f"[Logger] Обнаружен Google Colab, логи будут в {self.log_dir}")
        
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
            # ИСПРАВЛЕНИЕ: заменить plot_progress() на plot_progress_colab()
            self.plot_progress_colab()  # Используем метод для Colab
    
    def plot_progress_colab(self):
        """Специальный метод для отображения графиков в Colab"""
        if not self.train_losses:
            return
        
        # Создаем упрощенный график для быстрого просмотра в Colab
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 1. График потерь
        axes[0].plot(self.train_losses, alpha=0.7, linewidth=1)
        axes[0].set_xlabel('Шаги')
        axes[0].set_ylabel('Потери обучения')
        axes[0].set_title(f'Потери обучения (шаг {self.steps})')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Сглаженные потери
        if len(self.train_losses) > 10:
            window = min(50, len(self.train_losses) // 4)
            moving_avg = pd.Series(self.train_losses).rolling(window=window).mean()
            axes[1].plot(moving_avg, label=f'Среднее (окно={window})', 
                        color='red', linewidth=2)
            axes[1].plot(self.train_losses, alpha=0.2, label='Сырые значения')
            axes[1].set_xlabel('Шаги')
            axes[1].set_ylabel('Потери')
            axes[1].set_title('Сглаженные потери')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Прогресс обучения (Эпоха {self.epochs}, Шаги {self.steps})', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Показываем в Colab
        plt.show()
        
        # Также сохраняем
        plot_path = os.path.join(self.experiment_path, f'training_progress_colab_epoch_{self.epochs}.png')
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
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

    def display_colab_info(self):
        """Отображение информации для пользователей Colab"""
        if self.is_colab:
            print("\n" + "="*60)
            print("ИНФОРМАЦИЯ ДЛЯ GOOGLE COLAB:")
            print("="*60)
            print(f"📊 Графики обучения: {self.experiment_path}/")
            print(f"📁 Логи TensorBoard: {self.experiment_path}")
            print("="*60)
            print("\nЧтобы запустить TensorBoard в Colab, выполните:")
            print(f"  %load_ext tensorboard")
            print(f"  %tensorboard --logdir {self.experiment_path}")
            print("="*60 + "\n")    
    
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
        self.is_colab = 'google.colab' in sys.modules
        self.last_log_time = datetime.now()
        self.progress_interval = 50  # Шагов между выводом прогресса в Colab
    
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
                
                # Дополнительный вывод прогресса в Colab
                if self.is_colab and state.global_step % self.progress_interval == 0:
                    current_time = datetime.now()
                    elapsed = (current_time - self.last_log_time).total_seconds()
                    
                    print(f"\n[Шаг {state.global_step}] Потеря: {loss:.4f} | "
                          f"LR: {learning_rate:.2e} | "
                          f"Время с последнего: {elapsed:.1f}с")
                    
                    self.last_log_time = current_time
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце эпохи"""
        epoch = state.epoch
        train_loss = state.log_history[-1].get('loss') if state.log_history else None
        
        self.logger.log_epoch(
            epoch=int(epoch),
            train_loss=train_loss
        )
        
        # Специальный вывод для Colab
        if self.is_colab:
            print(f"\n{'='*50}")
            print(f"ЭПОХА {int(epoch)} ЗАВЕРШЕНА")
            print(f"{'='*50}")
            if train_loss:
                print(f"Средняя потеря за эпоху: {train_loss:.4f}")
            print(f"Всего шагов: {state.global_step}")
            print(f"Текущий learning rate: {args.learning_rate:.2e}")
            print(f"{'='*50}\n")
def save_to_drive_in_colab(output_dir, experiment_path):
    """Сохраняет результаты в Google Drive (только в Colab)"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        drive_path = '/content/drive/MyDrive/LLM_Training'
        os.makedirs(drive_path, exist_ok=True)
        
        # Копируем модель
        import shutil
        model_drive_path = os.path.join(drive_path, os.path.basename(output_dir))
        shutil.copytree(output_dir, model_drive_path, dirs_exist_ok=True)
        
        # Копируем логи
        logs_drive_path = os.path.join(drive_path, 'logs', os.path.basename(experiment_path))
        shutil.copytree(experiment_path, logs_drive_path, dirs_exist_ok=True)
        
        print(f"✅ Результаты сохранены в Google Drive:")
        print(f"   Модель: {model_drive_path}")
        print(f"   Логи: {logs_drive_path}")
        
        return True
    except Exception as e:
        print(f"⚠️ Не удалось сохранить в Google Drive: {e}")
        return False

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
        sep_text = tokenizer.eos_token
    else:
        # как fallback — добавим перенос строки (будет токенизирован)
        sep_token_id = None
        sep_text = "\n"

    # Отладочная статистика
    stats = {
        "total_examples": 0,
        "empty_responses": 0,
        "truncated_prompts": 0,
        "all_ignored_labels": 0,
        "avg_trainable_tokens": 0,
    }
    
    # Диагностика первых N примеров
    debug_first_n = min(5, len(items))
    print(f"\n🔍 ДИАГНОСТИКА первых {debug_first_n} примеров:")
    print("-" * 50)

    for idx, it in enumerate(items):
        prompt = it.get('context', '') or it.get('instruction', '') or ''
        response = it.get('utterance', '') or it.get('output', '') or it.get('response', '') or ''
        
        stats["total_examples"] += 1
        
        # Проверка на пустой response
        if not response.strip():
            stats["empty_responses"] += 1
            if idx < debug_first_n:
                print(f"Пример #{idx}: ПУСТОЙ response, пропускаем...")
            continue
            
        # токенизируем отдельно без добавления специальных токенов
        enc_prompt = tokenizer(prompt, add_special_tokens=False)
        enc_resp = tokenizer(response, add_special_tokens=False)

        prompt_ids = enc_prompt["input_ids"]
        resp_ids = enc_resp["input_ids"]
        
        # Проверка: response токенизировался в пустоту?
        if len(resp_ids) == 0:
            if idx < debug_first_n:
                print(f"Пример #{idx}: Response токенизировался в пустоту!")
                print(f"  Response текст: '{response[:50]}...'")
                print(f"  Токены: {tokenizer.tokenize(response[:50])}")
            continue

        # составляем input_ids: prompt + [sep?] + response
        if sep_token_id is not None:
            input_ids = prompt_ids + [sep_token_id] + resp_ids
            prompt_len = len(prompt_ids) + 1
        else:
            # если нет eos_token_id — положим явный разделитель как перевод строки и токенизируем
            sep_enc = tokenizer(sep_text, add_special_tokens=False)
            sep_ids = sep_enc["input_ids"]
            input_ids = prompt_ids + sep_ids + resp_ids
            prompt_len = len(prompt_ids) + len(sep_ids)

        original_length = len(input_ids)
        
        # ВАЖНОЕ ИСПРАВЛЕНИЕ: усекаем до max_length С КОНЦА (чтобы сохранить prompt)
        if len(input_ids) > max_length:
            # Обрезаем с конца, сохраняя prompt
            if prompt_len <= max_length:
                # prompt помещается полностью, обрезаем только response
                input_ids = input_ids[:max_length]
            else:
                # prompt слишком длинный, приходится обрезать и его
                input_ids = input_ids[:max_length]
                # Пересчитываем prompt_len для обрезанной версии
                # Находим, где заканчивается prompt в обрезанной последовательности
                # Это минимум между исходным prompt_len и max_length
                prompt_len = min(prompt_len, max_length)
                stats["truncated_prompts"] += 1
        
        # РАСЧЕТ labels: -100 для prompt, реальные id для response
        if prompt_len >= len(input_ids):
            # Случай 1: prompt занимает всю последовательность или больше
            labels = [-100] * len(input_ids)
            if idx < debug_first_n:
                print(f"Пример #{idx}: prompt занимает всю последовательность!")
        else:
            # Случай 2: есть response для обучения
            labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Проверка: все ли labels = -100?
        trainable_tokens = sum(1 for label in labels if label != -100)
        if trainable_tokens == 0:
            stats["all_ignored_labels"] += 1
        
        stats["avg_trainable_tokens"] += trainable_tokens
        
        # Диагностика первых примеров
        if idx < debug_first_n:
            num_ignored = len(labels) - trainable_tokens
            print(f"\nПример #{idx}:")
            print(f"  Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            print(f"  Response: '{response[:50]}{'...' if len(response) > 50 else ''}'")
            print(f"  Длина prompt: {len(prompt_ids)} токенов")
            print(f"  Длина response: {len(resp_ids)} токенов")
            print(f"  Исходная длина: {original_length}, после усечения: {len(input_ids)}")
            print(f"  prompt_len: {prompt_len}")
            print(f"  Игнорируемых токенов: {num_ignored}")
            print(f"  Обучаемых токенов: {trainable_tokens}")
            
            if trainable_tokens == 0:
                print(f"  ⚠️  ВНИМАНИЕ: Нет токенов для обучения!")
                if prompt_len >= len(input_ids):
                    print(f"    Причина: prompt_len ({prompt_len}) >= длина последовательности ({len(input_ids)})")
                elif len(resp_ids) == 0:
                    print(f"    Причина: response токенизировался в пустоту")
            else:
                # Покажем первые 10 обучаемых токенов
                trainable_positions = [i for i, label in enumerate(labels) if label != -100]
                first_trainable = trainable_positions[:5]
                print(f"  Первые обучаемые позиции: {first_trainable}")
        
        # attention_mask — все 1 (будет паддиться в collator)
        attention_mask = [1] * len(input_ids)

        records.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    # Вывод статистики
    print(f"\n📊 СТАТИСТИКА ПОДГОТОВКИ ДАННЫХ:")
    print(f"  Всего примеров: {stats['total_examples']}")
    print(f"  Пустых responses: {stats['empty_responses']}")
    print(f"  Усеченных prompts: {stats['truncated_prompts']}")
    print(f"  Примеров без обучаемых токенов: {stats['all_ignored_labels']}")
    
    if stats['total_examples'] - stats['all_ignored_labels'] > 0:
        avg_trainable = stats['avg_trainable_tokens'] / (stats['total_examples'] - stats['all_ignored_labels'])
        print(f"  Среднее обучаемых токенов на пример: {avg_trainable:.1f}")
    
    # Критическая проверка
    if stats['all_ignored_labels'] == stats['total_examples']:
        print(f"\n🚨 КРИТИЧЕСКАЯ ОШИБКА: Во всех примерах нет обучаемых токенов!")
        print(f"   Проверьте:")
        print(f"   1. Формат данных (ключи 'context', 'utterance', 'response', 'output')")
        print(f"   2. Что response не пуст")
        print(f"   3. Что prompt не слишком длинный (max_length={max_length})")
        return None, None
    
    if stats['all_ignored_labels'] > stats['total_examples'] * 0.5:
        print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Более 50% примеров без обучаемых токенов!")
        print(f"   Обучение может быть неэффективным.")

    # Разделяем на train и validation
    if len(records) == 0:
        print(f"\n❌ Нет данных для обучения!")
        return None, None
        
    total_samples = len(records)
    split_idx = int(total_samples * (1 - test_size))
    
    train_records = records[:split_idx]
    val_records = records[split_idx:] if test_size > 0 else []
    
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records) if val_records else None
    
    print(f"\n✅ Создано {len(train_records)} train и {len(val_records)} validation примеров")
    
    # Дополнительная проверка первых примеров из тренировочного набора
    if train_ds and len(train_ds) > 0:
        print(f"\n🔍 Проверка первого train примера:")
        sample = train_ds[0]
        print(f"  Длина input_ids: {len(sample['input_ids'])}")
        print(f"  Длина labels: {len(sample['labels'])}")
        trainable = sum(1 for label in sample['labels'] if label != -100)
        print(f"  Обучаемых токенов: {trainable}")
        
        if trainable > 0:
            # Покажем некоторые обучаемые токены
            positions = [i for i, label in enumerate(sample['labels']) if label != -100][:10]
            print(f"  Позиции обучаемых токенов: {positions}")
            
            # Декодируем небольшой фрагмент response
            response_start = positions[0] if positions else len(sample['labels']) - 5
            response_tokens = sample['input_ids'][response_start:response_start+10]
            try:
                decoded = tokenizer.decode(response_tokens, skip_special_tokens=True)
                print(f"  Пример response: '{decoded[:100]}...'")
            except:
                pass
    
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
    parser.add_argument('--save_to_drive', action='store_true', 
                       help='Сохранить результаты в Google Drive (только для Colab)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='Количество шагов для накопления градиента')
    parser.add_argument('--warmup_steps', type=int, default=100, 
                       help='Количество шагов для прогрева')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                       help='Максимальная норма градиента для обрезки')
    parser.add_argument('--optimizer', type=str, default='adamw_torch', 
                       choices=['adamw_torch', 'adamw_apex_fused', 'adafactor'],
                       help='Оптимизатор для обучения')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'constant', 'constant_with_warmup'],
                       help='Тип планировщика learning rate')

    args = parser.parse_args()
    
    # Проверяем, в Colab ли мы
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        print("="*60)
        print("🚀 ЗАПУСК В GOOGLE COLAB")
        print("="*60)
        
        # Автоматически настраиваем пути для Colab
        if args.log_dir == "./logs":
            args.log_dir = "/content/logs"
        if args.output_dir == "./fine-tuned-lora":
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output_dir = f"/content/fine-tuned-lora-{timestamp}"
        
        print(f"📁 Логи: {args.log_dir}")
        print(f"💾 Выходная директория: {args.output_dir}")
        print("="*60)
    
    # Инициализация логгера
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    logger.display_colab_info()
    
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
        'eval_steps': args.eval_steps,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_steps': args.warmup_steps,
        'optimizer': args.optimizer,
        'lr_scheduler_type': args.lr_scheduler_type,
        'max_grad_norm': args.max_grad_norm
    }
    
    print(f"\n📋 Конфигурация обучения:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)
    
    # Устанавливаем сиды для воспроизводимости
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"\n📥 Загрузка токенизатора и модели: {args.model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        
        # Гарантируем наличие pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"✅ Используем eos_token ({tokenizer.eos_token}) как pad_token")
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                print("✅ Добавлен pad_token: [PAD]")
        
        # Проверяем наличие специальных токенов
        print(f"📝 Токенизатор настроен:")
        print(f"   pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"   eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"   bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
        
        # Загружаем базовую модель
        print(f"🤖 Загрузка модели {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True  # Для нестандартных архитектур
        )
        
        # Для тренировки выключаем кэш генерации
        try:
            model.gradient_checkpointing_enable()
            print("✅ Градиентный чекпоинт активирован")
        except Exception as e:
            print(f"⚠️ Не удалось включить gradient checkpointing: {e}")
        
        model.config.use_cache = False
        
        # Настройка LoRA
        if args.use_lora:
            target_modules = get_default_lora_targets(args.model_name)
            print(f"\n🎯 Используем target_modules для LoRA: {target_modules}")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            # покажем сколько параметров обучается
            model.print_trainable_parameters()
        else:
            print("⚠️ LoRA отключен, обучаются все параметры модели")
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели/токенизатора: {e}")
        logger.close()
        return
    
    # Определяем тип данных и загружаем
    print(f"\n📊 Загрузка данных из {args.data_path}")
    
    if not os.path.exists(args.data_path):
        print(f"❌ Файл {args.data_path} не найден!")
        logger.close()
        return
    
    try:
        if args.data_path.endswith('.md'):
            train_ds, eval_ds = prepare_markdown_dataset(
                text=read_markdown(args.data_path),
                tokenizer=tokenizer,
                max_length=args.max_length,
                test_size=0 if args.no_eval else args.test_size
            )
        else:
            items = read_json_or_jsonl(args.data_path)
            print(f"📄 Загружено {len(items)} записей из файла")
            if items:
                # Покажем пример данных
                print("\n📋 Пример первой записи:")
                for key in items[0].keys():
                    print(f"   {key}: {items[0][key][:100]}...")
            
            train_ds, eval_ds = prepare_dataset(
                items, 
                tokenizer, 
                args.max_length,
                test_size=0 if args.no_eval else args.test_size
            )
        
        print(f"\n📊 Размер тренировочного датасета: {len(train_ds)} примеров")
        if eval_ds:
            print(f"📊 Размер валидационного датасета: {len(eval_ds)} примеров")
            
            # Покажем пример токенизированных данных
            print("\n🔍 Пример токенизированных данных:")
            sample = train_ds[0]
            print(f"   input_ids длина: {len(sample['input_ids'])}")
            print(f"   labels: {sample['labels'][:20]}...")
            print(f"   Не игнорируемых токенов: {sum(1 for x in sample['labels'] if x != -100)}")
        else:
            print("⚠️ Валидационный датасет не создан")
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        logger.close()
        return
    
    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    # Определяем стратегию валидации
    if args.no_eval or args.eval_steps <= 0 or eval_ds is None:
        eval_strategy = "no"
        eval_steps = None
        load_best_model_at_end = False
        print("\n⚠️ Валидация отключена")
    else:
        eval_strategy = "steps"
        eval_steps = args.eval_steps
        load_best_model_at_end = True
        print(f"\n✅ Валидация включена каждые {eval_steps} шагов")

    # Настройка аргументов обучения
    print(f"\n⚙️ Настройка параметров обучения...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size * 2,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy=eval_strategy,
        save_total_limit=3,
        fp16=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="loss" if eval_ds else None,
        greater_is_better=False,
        logging_dir=logger.experiment_path,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        group_by_length=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        eval_accumulation_steps=None,
        prediction_loss_only=True,
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

    print(f"\n🚀 Начинаем обучение...")
    print(f"   Всего шагов: {len(train_ds) * args.num_train_epochs // (args.per_device_train_batch_size * args.gradient_accumulation_steps)}")
    print(f"   Графики в реальном времени: {logger.experiment_path}/")
    print("-" * 60)
    
    try:
        # Сохраняем конфигурацию
        config_file = os.path.join(logger.experiment_path, 'training_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"📝 Конфигурация сохранена в: {config_file}")

        print("\n🔍 ПРОВЕРКА DATA COLLATOR:")
        # Проверим несколько примеров из датасета
        sample_batch = [train_ds[i] for i in range(2)]
        collated = data_collator(sample_batch)

        print(f"input_ids shape: {collated['input_ids'].shape}")
        print(f"labels shape: {collated['labels'].shape}")

        # Проверим, что labels не все -100
        print(f"Уникальные значения в labels: {torch.unique(collated['labels'])[:10].tolist()}")
        print(f"Количество -100 в первом примере: {(collated['labels'][0] == -100).sum().item()}")
        print(f"Количество реальных токенов в первом примере: {(collated['labels'][0] != -100).sum().item()}")

        # Проверим типы данных
        print(f"\nТипы данных:")
        print(f"  input_ids dtype: {collated['input_ids'].dtype}")
        print(f"  labels dtype: {collated['labels'].dtype}")
        print(f"  attention_mask dtype: {collated['attention_mask'].dtype}")
        
        train_result = trainer.train()
        
        # Сохраняем метрики обучения
        metrics = train_result.metrics
        metrics_file = os.path.join(logger.experiment_path, 'training_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Метрики обучения сохранены в: {metrics_file}")
        print(f"📊 Финальные метрики: {metrics}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
        
        # Создаем промежуточный отчет
        logger.plot_final_summary(args.model_name, config)
        
        # Сохраняем прерванную модель
        interrupted_dir = args.output_dir + "_interrupted"
        os.makedirs(interrupted_dir, exist_ok=True)
        
        if args.use_lora:
            model.save_pretrained(interrupted_dir)
            tokenizer.save_pretrained(interrupted_dir)
            print(f"💾 Промежуточная модель сохранена в: {interrupted_dir}")
        
        logger.close()
        
        # Копируем логи
        import shutil
        logs_in_output = os.path.join(interrupted_dir, 'training_logs')
        shutil.copytree(logger.experiment_path, logs_in_output, dirs_exist_ok=True)
        
        if args.save_to_drive and is_colab:
            save_to_drive_in_colab(interrupted_dir, logger.experiment_path)
        
        print("Обучение остановлено.")
        return
        
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
        
        # Пытаемся сохранить хоть что-то
        error_dir = args.output_dir + "_error"
        os.makedirs(error_dir, exist_ok=True)
        
        # Сохраняем информацию об ошибке
        error_info = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        error_file = os.path.join(logger.experiment_path, 'error_info.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        logger.close()
        raise
    
    finally:
        # Всегда закрываем логгер
        logger.close()

    print(f"\n💾 Сохраняем результат в {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Если использовали LoRA: сохраняем ТОЛЬКО адаптер
    if args.use_lora:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"✅ Сохранён LoRA-адаптер и токенизатор в {args.output_dir}")
        
        # Также сохраняем конфигурацию LoRA
        lora_config_file = os.path.join(args.output_dir, 'adapter_config.json')
        if os.path.exists(lora_config_file):
            with open(lora_config_file, 'r', encoding='utf-8') as f:
                lora_cfg = json.load(f)
            print(f"📄 Конфигурация LoRA: r={lora_cfg.get('r')}, alpha={lora_cfg.get('lora_alpha')}")
    else:
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"✅ Сохранена полная модель и токенизатор в {args.output_dir}")

    # Создаем финальный отчет
    print("\n📈 Создаем финальный отчет...")
    logger.plot_final_summary(args.model_name, config)
    
    # Копируем логи в output_dir
    import shutil
    logs_in_output = os.path.join(args.output_dir, 'training_logs')
    shutil.copytree(logger.experiment_path, logs_in_output, dirs_exist_ok=True)
    print(f"📊 Логи обучения также сохранены в: {logs_in_output}")

    # Сохраняем в Google Drive если нужно
    if args.save_to_drive and is_colab:
        print("\n☁️ Сохраняем в Google Drive...")
        if save_to_drive_in_colab(args.output_dir, logger.experiment_path):
            print("✅ Успешно сохранено в Google Drive!")
        else:
            print("⚠️ Не удалось сохранить в Google Drive")

    # Показываем итоговую информацию
    print("\n" + "="*60)
    print("🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print("="*60)
    print(f"📊 Результаты сохранены в:")
    print(f"   Модель: {args.output_dir}")
    print(f"   Логи: {logs_in_output}")
    print(f"   Графики: {logger.experiment_path}/")
    
    if eval_ds:
        print(f"\n📈 Финальные метрики:")
        print(f"   Лучшая валидационная потеря: {min(logger.val_losses) if logger.val_losses else 'N/A':.4f}")
    
    print(f"\n📊 Статистика обучения:")
    print(f"   Всего шагов: {logger.steps}")
    print(f"   Всего эпох: {logger.epochs}")
    print(f"   Всего параметров (обучаемых): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Показываем путь к последнему графику
    if is_colab:
        latest_plot = os.path.join(logger.experiment_path, 'final_training_summary.png')
        if os.path.exists(latest_plot):
            print(f"\n📸 Финальный график: {latest_plot}")
            
            # Показываем график прямо в выводе (только в Colab)
            try:
                from IPython.display import Image, display
                print("\n📊 Визуализация финальных результатов:")
                display(Image(filename=latest_plot))
            except:
                print("(График доступен по указанному пути)")
    
    print("="*60)
    print("Готово! ✅")


if __name__ == '__main__':
    main()
