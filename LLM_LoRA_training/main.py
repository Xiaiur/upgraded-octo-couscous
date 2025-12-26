import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel


# -------------------- Утилиты --------------------
def read_markdown(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def prepare_markdown_dataset(
    text: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    stride: int = 128
):
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
            "labels": chunk.copy(),  # ← ВАЖНО
        })

    return Dataset.from_list(records)

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
    # Для Qwen — имена модулей могут отличаться; указаны общие варианты.
    # При необходимости поправьте под конкретную архитектуру (с помощью просмотра model.named_modules()).
    if 'llama' in mn or 'alpaca' in mn:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if 'gpt2' in mn or 'gpt' in mn or 'dialo' in mn:
        return ["c_attn", "c_proj", "c_fc", "c_ffn"]
    # Qwen/Falcon/иные: стандартная попытка
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


# -------------------- Подготовка данных --------------------

def prepare_dataset(items: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 512):
    """
    Корректно токенизируем prompt и response отдельно, собираем input_ids и labels
    labels: -100 для токенов prompt (и sep), реальные id для токенов response
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

    ds = Dataset.from_list(records)
    return ds


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

    parser.add_argument('--model_name', type=str, default='gpt2', help='Название модели в HF (пример: gpt2, microsoft/DialoGPT-medium, Qwen/Qwen2.5-1.5B и т.д.)')
    parser.add_argument('--data_path', type=str, required=True, help='Путь к data.json или data.jsonl')
    parser.add_argument('--output_dir', type=str, default='./fine-tuned-lora', help='Куда сохранить модель/адаптер')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    print(f"Загрузка токенизатора и модели: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Гарантируем наличие pad_token
    if tokenizer.pad_token is None:
        # Лучше сделать pad_token == eos_token, если у модели нет отдельного pad
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
        
    text = read_markdown(args.data_path)
    ds = prepare_markdown_dataset(
        text=text,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    # ВАЖНО: отключаем автоматическое сохранение чекпоинтов (они большие).
    # После тренировки вручную сохраним только адаптер LoRA (если был).
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="no",           # **не сохранять чекпоинты Trainer'а**
        save_total_limit=1,
        fp16=True,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],
        # при необходимости: gradient_accumulation_steps, warmup_steps и др.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    print("Начинаем обучение...")
    trainer.train()

    print("Сохраняем результат...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Если использовали LoRA: сохраняем ТОЛЬКО адаптер (маленькая папка)
    if args.use_lora:
        # model — это PeftModel, .save_pretrained() сохранит adapter weights (а не полную модель)
        model.save_pretrained(args.output_dir)
        # НЕ сохраняем базовую модель в этой папке, чтобы не захламлять диск.
        # При загрузке: сначала load base model, затем PeftModel.from_pretrained(...)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Сохранён LoRA-адаптер и токенизатор в {args.output_dir}")
    else:
        # если LoRA не использовался — сохраним полную модель (safe_serialization)
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Сохранена полная модель и токенизатор в {args.output_dir}")

    print("Готово.")


if __name__ == '__main__':
    main()
