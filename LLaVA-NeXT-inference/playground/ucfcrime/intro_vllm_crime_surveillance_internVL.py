def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.chat(inputs.input_ids, max_length=200, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


import torch
from transformers import AutoTokenizer, AutoModel
path = "/home/share/chenhaoran/model_zoo/models--OpenGVLab--InternVL2-40B/snapshots/b52a031c8dc5c9fc2da55daae3cf1d7062371d13/"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()

# pure-text conversation (纯文本对话)
generation_config = dict(max_new_tokens=64, do_sample=False)
question = 'Hello, who are you?'
response = model.chat(tokenizer, None, question, generation_config, history=None, return_history=False)
print(f'User: {question}\nAssistant: {response}')