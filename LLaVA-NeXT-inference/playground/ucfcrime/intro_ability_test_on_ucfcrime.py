import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

def load_model(path):
    if 'internvl' in path.lower():
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

    elif 'llava' in path.lower():
        from llava.model.builder import load_pretrained_model
        pretrained = "/home/share/chenhaoran/model_zoo/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd/"
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        config = AutoConfig.from_pretrained(pretrained)
        config.mm_vision_tower = "/home/share/models/siglip-so400m-patch14-384/"
        tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa",customized_config=config
        )
        model = model.to(device)
        model.eval()

    elif 'internvideo' in path.lower():
        tokenizer =  AutoTokenizer.from_pretrained(path,trust_remote_code=True,use_fast=False)
        model = AutoModel.from_pretrained(path,torch_dtype=torch.bfloat16,trust_remote_code=True).cuda()

    return
