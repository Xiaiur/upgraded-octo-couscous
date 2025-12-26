from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import argparse
import os

parser = argparse.ArgumentParser(description='Chatting with fine-tune causal LM with LoRA (PEFT)')
parser.add_argument('--model_name', type=str, required=True, help='Название модели(из HF)')
parser.add_argument('--model_dir', type=str, help='Директория с моделью')
parser.add_argument('--test_seed', type=int, required=False, help='Сид для генерации воспроизводимых ответов ')

args = parser.parse_args()

if args.test_seed is not None:
    set_seed(args.test_seed)
    import torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.test_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def chat(model, tokenizer):
    while True:
        prompt = input("Введите вопрос: ")
        if prompt == 'q':
            return

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.5, top_p=0.3, top_k=5)
        print(f"Ответ: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype="auto",
    device_map="auto",
)
if args.model_dir is not None:
    model = PeftModel.from_pretrained(model, args.model_dir)
    chat(model, tokenizer)
else:
    chat(model, tokenizer)