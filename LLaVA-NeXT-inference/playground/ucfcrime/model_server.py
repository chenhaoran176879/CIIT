from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Query(BaseModel):
    messages: list


model_path = "/home/share/chenhaoran/model_zoo/Qwen1.5-72B-Chat-GPTQ-Int4/"
device = "cuda" #if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

app = FastAPI()

@app.post("/generate/")
async def generate(query: Query):
    text = tokenizer.apply_chat_template(
        query.messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print('receive request:',text)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)


    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('response:',response)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
