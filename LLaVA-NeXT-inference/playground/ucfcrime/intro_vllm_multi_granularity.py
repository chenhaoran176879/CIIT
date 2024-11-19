from UCVLDataset import UCVLDataset
import argparse
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig,AutoProcessor
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token
import copy
from llava.conversation import conv_templates
from video_utils import load_video_internVL,load_video_llava,load_video_internvideo,load_last_json_index,qwen_process_vision_info

import os
import json



def preprocess(args,tokenizer,video_path,image_processor,prompt):
    model_path = args.model_path
    if 'internvl' in model_path.lower():
        pixel_values, num_patches_list = load_video_internVL(video_path, num_segments=args.nframes, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt
        return {
            "pixel_values": pixel_values,
            "question": question,
            "num_patches_list": num_patches_list
        }

    elif 'llava' in model_path.lower():
        video_frames = load_video_llava(video_path, args.nframes,)# start_time=video_meta['start_time'], end_time=video_meta['end_time'])
        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
        image_sizes = [frame.size for frame in video_frames]
        modalities = ["video"] * len(video_frames)

        return {"input_ids":input_ids,
                "image_tensors":image_tensors,
                "image_sizes":image_sizes,
                "modalities":modalities
                }

    elif 'internvideo' in model_path.lower():
        video_tensor = load_video_internvideo(video_path, num_segments=args.nframes, return_msg=False, resolution=224, hd_num=6)
        video_tensor = video_tensor
        return {"video_tensor":video_tensor,"prompt":prompt}

    elif 'qwen' in model_path.lower() and 'vl' in model_path.lower():
        processor = AutoProcessor.from_pretrained(model_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "nframes":args.nframes
                    },
                    {"type": "text",
                     "text": prompt},
                ],
            }
            ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = qwen_process_vision_info(messages)
        
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        return {"inputs":inputs}



def get_model(model_path):
    if 'internvl' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
        return model,None
    
    elif 'llava' in model_path.lower():
        from llava.model.builder import load_pretrained_model
        pretrained = model_path
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        print("loading llava")
        config = AutoConfig.from_pretrained(pretrained)
        config.mm_vision_tower = "/home/share/models/siglip-so400m-patch14-384/"
        tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa",customized_config=config
        )
        model = model.to(device)
        model.eval()
        return model,image_processor
    
    elif 'internvideo' in model_path.lower():
        model = AutoModel.from_pretrained(model_path,torch_dtype=torch.bfloat16,trust_remote_code=True).cuda()
        return model,None
    

    elif 'qwen' in model_path.lower() and 'vl' in model_path.lower():
        from transformers import Qwen2VLForConditionalGeneration
        print("list model path:\n",os.listdir(model_path))
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        return model,None


def model_generate(model_path,model,inputs,tokenizer):
    if 'internvl' in model_path.lower():
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response, history = model.chat(tokenizer, inputs['pixel_values'], inputs['question'], generation_config,
                               num_patches_list=inputs['num_patches_list'], history=None, return_history=True)
        return response
    
    
    elif 'llava' in model_path.lower():
        cont = model.generate(
            inputs['input_ids'],
            images=inputs['image_tensors'],
            image_sizes=inputs['image_sizes'],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=inputs['modalities']
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs

    elif 'internvideo' in model_path.lower():
        chat_history = []
        response, chat_history = model.chat(tokenizer, '', 
                                            'Describe the video step by step',
                                            instruction= inputs["prompt"],
                                            media_type='video', 
                                            media_tensor=inputs['video_tensor'], 
                                            chat_history= chat_history, 
                                            return_history=True,
                                            generation_config={'do_sample':True,'max_new_tokens':512,})
        return response

    elif 'qwen' in model_path.lower() and 'vl' in model_path.lower():
        inputs = inputs['inputs']
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        print("generation complete")

        generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        processor = AutoProcessor.from_pretrained(model_path)
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if len(output_text) == 1 and isinstance(output_text,list):
            output_text = output_text[0]

        return output_text



    
def eval_UCVL(args,dataset,model,tokenizer,image_processor):
    model_name = os.path.basename(args.model_path)
    start_index = load_last_json_index(args.output_jsonl,collect_scores=False)
    output_f = open(args.output_jsonl, 'a', encoding='utf-8')
    print(f"Starting evaluation for model {model_name} from index: {start_index}")

    with torch.no_grad():
        for idx in range(start_index,len(dataset)):
            data = dataset[idx]  # 获取数据
            print(f"Processing index: {idx}, video name: {data['video_name']}")
            answer_cache = {}
            print("********data**********\n",data)
            for question_name, question_prompt in data['bench_questions'].items():
                print(f"  Question: {question_name}")
                print(question_prompt)
                inputs = preprocess(args,
                                    tokenizer=tokenizer,
                                    video_path=data['video_path'],
                                    image_processor=image_processor,
                                    prompt=question_prompt)

                output_text = model_generate(model_path=args.model_path,
                                             model=model,
                                             inputs=inputs,
                                             tokenizer=tokenizer)

                # 将问题名称和对应的答案收集到字典中
                answer_cache[question_name] = output_text
                print(f"  Answer for '{question_name}': {output_text}")

            answer = {
                "index":idx,
                "data_split":data['data_split'],
                "model_name":model_name,
                "video_name": data['video_name'],
                "nframes": args.nframes,
                "answer":answer_cache
            }
            output_f.write(json.dumps(answer, ensure_ascii=False) + '\n')
            del inputs, output_text, answer_cache, answer
        
        output_f.close()



                


def main():
    
    #video_folder = "/mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/"
    parser = argparse.ArgumentParser(description="UCVLDataset Setup")

    # 添加参数
    parser.add_argument('--jsonl_root', type=str, required=True, 
                        help="Path to the combined JSONL file")

    parser.add_argument('--output_jsonl', type=str, required=True, 
                        help="Path to the output JSONL file")
    
    parser.add_argument('--video_folder', type=str, required=True, 
                        help="Path to the root video folder")

    parser.add_argument('--nframes', type=int, default=32, 
                        help="num frames sampled from a video")

    parser.add_argument('--max_nframes', type=int, default=100, 
                        help="max num frames sampled from a video")
    
    parser.add_argument('--min_nframes', type=int, default=32, 
                        help="min num frames sampled from a video")
    
    parser.add_argument('--model_path', type=str, required=True, 
                        help="Path to the model file")
    parser.add_argument('--data_split', type=str, default='train', choices=['train', 'val', 'test'],
                        help="Which dataset split to use (train, val, test)")
    
    parser.add_argument('--questions', nargs='+', default=[
                        'anomaly_detection_question',
                        'event_description_question',
                        'crime_classification_question',
                        'event_description_with_classification',
                        'temporal_grounding_question',
                        'multiple_choice_question'],
                        choices=[
                        'anomaly_detection_question',
                        'event_description_question',
                        'crime_classification_question',
                        'event_description_with_classification',
                        'temporal_grounding_question',
                        'multiple_choice_question'],
                        help="Select questions to answer (choose multiple) from \
                        'anomaly_detection_question', \
                        'event_description_question', \
                        'crime_classification_question', \
                        'event_description_with_classification', \
                        'temporal_grounding_question', \
                        'multiple_choice_question'")
    
    # 解析参数
    args = parser.parse_args()
    print(f"Selected questions: {', '.join(args.questions)}")
    dataset = UCVLDataset(json_root=args.jsonl_root,video_folder=args.video_folder,questions=args.questions,split=args.data_split)
    model,image_processor = get_model(model_path=args.model_path)
    model.eval()
    tokenizer = None
    if 'qwen2-vl' not in args.model_path:
        print('loading tokenizer')
        tokenizer =  AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True,use_fast=False)
    eval_UCVL(args,dataset,model,tokenizer,image_processor)




if __name__ == "__main__":
    main()
'''
python -u /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_UCVL.py \
       --jsonl_root /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/LLM_summarization_results_all_with_normal.jsonl \
       --video_folder /mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/ \
       --nframes 32 \ 
       --model_path /home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-8B \
       --data_split test \
       --output_jsonl /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_internvl2-8B.jsonl
'''