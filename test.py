from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B"
lora_path = "./my-lora-model"   # путь к папке с adapter_model.bin

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto",
)

model = PeftModel.from_pretrained(model, lora_path)

prompts = [
    "Искусственный интеллект — это",
    "Обучение с учителем заключается в том, что",
    "Квантовые компьютеры отличаются от классических тем, что",
]
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

