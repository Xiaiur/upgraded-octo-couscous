import argparse
import os
import csv
from pathlib import Path
from typing import List, Tuple
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_prompts_from_txt(file_path: str) -> List[str]:
    """Загрузить промпты из TXT файла"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Пропускаем пустые строки и комментарии
                prompts.append(line)
    return prompts


def load_prompts_from_csv(file_path: str, column_name: str = 'prompt') -> List[str]:
    """Загрузить промпты из CSV файла"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if column_name not in reader.fieldnames:
            available = ', '.join(reader.fieldnames)
            raise ValueError(f"Колонка '{column_name}' не найдена. Доступные колонки: {available}")
        
        for row in reader:
            if row[column_name] and row[column_name].strip():
                prompts.append(row[column_name].strip())
    return prompts


def load_prompts(args) -> List[str]:
    """Загрузить промпты из различных источников"""
    
    # Если указан файл с промптами
    if args.prompts_file:
        file_path = Path(args.prompts_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {args.prompts_file}")
        
        if file_path.suffix.lower() == '.txt':
            print(f"📄 Загрузка промптов из TXT файла: {args.prompts_file}")
            return load_prompts_from_txt(args.prompts_file)
        
        elif file_path.suffix.lower() == '.csv':
            print(f"📊 Загрузка промптов из CSV файла: {args.prompts_file}")
            return load_prompts_from_csv(args.prompts_file, args.csv_column)
        
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}. Используйте .txt или .csv")
    
    # Если промпты переданы напрямую через командную строку
    elif args.prompts:
        print(f"📝 Использование промптов из командной строки")
        return args.prompts
    
    # Если ничего не указано, используем примерные промпты из D&D
    else:
        print("⚠️  Файл с промптами не указан, используются примерные промпты по D&D 3.5")
        return [
            "How does Bless work in D&D 3.5?",
            "I attack the zombie with my longsword",
            "Can I cast Fireball as a 5th level wizard?",
            "What's the DC for a Perception check?",
            "How much damage does a greatsword do on a critical hit?"
        ]


def load_models(model_name: str, lora_path: str = None) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM]:
    """
    Загружает базовую и дообученную модели
    
    Returns:
        tuple: (base_model, fine_tuned_model)
    """
    print(f"🤖 Загрузка базовой модели: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Загружаем базовую модель
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # Загружаем дообученную модель (с LoRA)
    if lora_path and os.path.exists(lora_path):
        print(f"🔗 Загрузка LoRA адаптера: {lora_path}")
        fine_tuned_model = PeftModel.from_pretrained(base_model, lora_path)
        print("✅ Модели загружены успешно")
        return base_model, fine_tuned_model, tokenizer
    else:
        print("⚠️  LoRA адаптер не найден, используется только базовая модель")
        return base_model, base_model, tokenizer


def generate_response(model, tokenizer, prompt: str, gen_args: dict) -> str:
    """Сгенерировать ответ от модели"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_args
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Убираем повтор промпта из ответа для чистоты
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response


def compare_models(base_model, fine_tuned_model, tokenizer, prompts: List[str], 
                   gen_args: dict, output_file: str = None):
    """Сравнить ответы базовой и дообученной моделей"""
    
    results = []
    total_base_time = 0
    total_finetuned_time = 0
    
    print("=" * 100)
    print(f"{'📊 СРАВНЕНИЕ МОДЕЛЕЙ':^100}")
    print("=" * 100)
    print(f"{'ПАРАМЕТРЫ ГЕНЕРАЦИИ:':<30} max_tokens={gen_args['max_new_tokens']}, "
          f"temp={gen_args['temperature']}, top_p={gen_args['top_p']}")
    print("=" * 100)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'🚀 ПРОМПТ ' + str(i) + '/' + str(len(prompts)) + ' ':─^100}")
        print(f"📝: {prompt}")
        print("-" * 100)
        
        # Генерация от базовой модели
        print(f"{'🔵 БАЗОВАЯ МОДЕЛЬ':<50}{'🟢 ДООБУЧЕННАЯ МОДЕЛЬ':<50}")
        print("-" * 100)
        
        start_time = time.time()
        base_response = generate_response(base_model, tokenizer, prompt, gen_args)
        base_time = time.time() - start_time
        total_base_time += base_time
        
        start_time = time.time()
        finetuned_response = generate_response(fine_tuned_model, tokenizer, prompt, gen_args)
        finetuned_time = time.time() - start_time
        total_finetuned_time += finetuned_time
        
        # Разбиваем ответы на строки для параллельного вывода
        base_lines = base_response.split('\n')
        finetuned_lines = finetuned_response.split('\n')
        max_lines = max(len(base_lines), len(finetuned_lines))
        
        # Выводим ответы параллельно
        for j in range(max_lines):
            base_line = base_lines[j] if j < len(base_lines) else ""
            finetuned_line = finetuned_lines[j] if j < len(finetuned_lines) else ""
            
            # Обрезаем длинные строки для лучшего отображения
            if len(base_line) > 45:
                base_line = base_line[:42] + "..."
            if len(finetuned_line) > 45:
                finetuned_line = finetuned_line[:42] + "..."
            
            print(f"{base_line:<50}{finetuned_line:<50}")
        
        print(f"{f'⏱️ {base_time:.2f}s':<50}{f'⏱️ {finetuned_time:.2f}s':<50}")
        print("-" * 100)
        
        # Сохраняем результаты
        results.append({
            'prompt': prompt,
            'base_response': base_response,
            'finetuned_response': finetuned_response,
            'base_time': base_time,
            'finetuned_time': finetuned_time
        })
    
    # Статистика
    print("\n" + "=" * 100)
    print(f"{'📈 СТАТИСТИКА':^100}")
    print("=" * 100)
    print(f"{'МЕТРИКА':<30} {'БАЗОВАЯ':<20} {'ДООБУЧЕННАЯ':<20} {'РАЗНИЦА':<20}")
    print("-" * 100)
    print(f"{'Среднее время ответа':<30} {total_base_time/len(prompts):<20.3f}s "
          f"{total_finetuned_time/len(prompts):<20.3f}s "
          f"{(total_finetuned_time - total_base_time)/len(prompts):<+20.3f}s")
    print(f"{'Общее время':<30} {total_base_time:<20.3f}s {total_finetuned_time:<20.3f}s "
          f"{total_finetuned_time - total_base_time:<+20.3f}s")
    
    # Сравнение длин ответов
    total_base_chars = sum(len(r['base_response']) for r in results)
    total_finetuned_chars = sum(len(r['finetuned_response']) for r in results)
    print(f"{'Суммарная длина ответов':<30} {total_base_chars:<20} chars "
          f"{total_finetuned_chars:<20} chars "
          f"{total_finetuned_chars - total_base_chars:<+20} chars")
    
    # Сохранение результатов в файл
    if output_file:
        save_results(results, output_file, gen_args)
    
    return results


def save_results(results: List[dict], output_file: str, gen_args: dict):
    """Сохранить результаты сравнения в файл"""
    import json
    from datetime import datetime
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'generation_parameters': gen_args,
        'total_prompts': len(results),
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Сравнение базовой и дообученной моделей',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python test_comparison.py --prompts_file prompts.txt
  python test_comparison.py --prompts "How does Bless work?" "I attack the orc"
  python test_comparison.py --lora_path ./my-lora-model --csv_column instruction
        """
    )
    
    # Обязательные параметры
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B",
                       help='Название базовой модели в HF (по умолчанию: Qwen/Qwen2.5-1.5B)')
    parser.add_argument('--lora_path', type=str, default="./my-lora-model",
                       help='Путь к папке с адаптером LoRA (по умолчанию: ./my-lora-model)')
    
    # Параметры для загрузки промптов
    parser.add_argument('--prompts_file', type=str, 
                       help='Путь к файлу с промптами (.txt или .csv)')
    parser.add_argument('--prompts', type=str, nargs='+',
                       help='Список промптов прямо в командной строке')
    parser.add_argument('--csv_column', type=str, default='prompt',
                       help='Название колонки в CSV файле (по умолчанию: prompt)')
    
    # Параметры генерации
    parser.add_argument('--max_new_tokens', type=int, default=150,
                       help='Максимальное количество новых токенов (по умолчанию: 150)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Температура для генерации (по умолчанию: 0.7)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling параметр (по умолчанию: 0.9)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling параметр (по умолчанию: 50)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                       help='Штраф за повторения (по умолчанию: 1.2)')
    parser.add_argument('--num_return_sequences', type=int, default=1,
                       help='Количество вариантов ответа (по умолчанию: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed для воспроизводимости (по умолчанию: 42)')
    
    # Дополнительные параметры
    parser.add_argument('--output_file', type=str, default="comparison_results.json",
                       help='Файл для сохранения результатов (по умолчанию: comparison_results.json)')
    parser.add_argument('--no_compare', action='store_true',
                       help='Не сравнивать, просто запустить дообученную модель')
    
    args = parser.parse_args()
    
    # Устанавливаем seed для воспроизводимости
    if args.seed is not None:
        import numpy as np
        import random
        
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Загружаем промпты
    try:
        prompts = load_prompts(args)
    except Exception as e:
        print(f"❌ Ошибка загрузки промптов: {e}")
        return
    
    if not prompts:
        print("❌ Нет промптов для тестирования")
        return
    
    print(f"🎯 Загружено промптов: {len(prompts)}")
    
    # Загружаем модели
    try:
        base_model, fine_tuned_model, tokenizer = load_models(args.model_name, args.lora_path)
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        return
    
    # Параметры генерации
    gen_args = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
        'num_return_sequences': args.num_return_sequences,
        'do_sample': True if args.temperature > 0 else False,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    
    if args.no_compare:
        # Просто запускаем дообученную модель
        print("\n" + "=" * 100)
        print(f"{'🚀 ТЕСТИРОВАНИЕ ДООБУЧЕННОЙ МОДЕЛИ':^100}")
        print("=" * 100)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Промпт {i}/{len(prompts)}: {prompt}")
            print("-" * 100)
            
            response = generate_response(fine_tuned_model, tokenizer, prompt, gen_args)
            print(f"🤖 Ответ: {response}")
            print("=" * 100)
    else:
        # Сравниваем модели
        results = compare_models(
            base_model=base_model,
            fine_tuned_model=fine_tuned_model,
            tokenizer=tokenizer,
            prompts=prompts,
            gen_args=gen_args,
            output_file=args.output_file
        )
        
        # Выводим итоговый анализ
        print("\n" + "=" * 100)
        print(f"{'📋 ИТОГОВЫЙ АНАЛИЗ':^100}")
        print("=" * 100)
        
        # Анализируем различия в ответах
        improvements = 0
        same = 0
        worse = 0
        
        for result in results:
            base_len = len(result['base_response'])
            finetuned_len = len(result['finetuned_response'])
            
            # Простой эвристический анализ
            dnd_keywords = ['spell', 'attack', 'damage', 'DC', 'roll', 'check', 'level', 'save']
            base_dnd_count = sum(1 for kw in dnd_keywords if kw in result['base_response'].lower())
            finetuned_dnd_count = sum(1 for kw in dnd_keywords if kw in result['finetuned_response'].lower())
            
            if finetuned_dnd_count > base_dnd_count:
                improvements += 1
            elif finetuned_dnd_count == base_dnd_count:
                same += 1
            else:
                worse += 1
        
        print(f"\n📊 КАЧЕСТВО ОТВЕТОВ (по наличию D&D терминов):")
        print(f"   🟢 Улучшено: {improvements}/{len(results)} ({improvements/len(results)*100:.1f}%)")
        print(f"   ⚪ Без изменений: {same}/{len(results)} ({same/len(results)*100:.1f}%)")
        print(f"   🔴 Ухудшено: {worse}/{len(results)} ({worse/len(results)*100:.1f}%)")
        
        # Примеры лучших улучшений
        if improvements > 0:
            print(f"\n🎯 Примеры улучшенных ответов:")
            for i, result in enumerate(results[:3]):  # Показываем первые 3
                print(f"   {i+1}. Промпт: {result['prompt'][:50]}...")
                print(f"      Базовая: {result['base_response'][:60]}...")
                print(f"      Дообученная: {result['finetuned_response'][:60]}...")
                print()


if __name__ == "__main__":
    main()
