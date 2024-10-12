import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig,AutoProcessor
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
import copy
from llava.conversation import conv_templates
from qwen_vl_utils import process_vision_info


# internvl's load video
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def HD_transform_padding(frames, image_size=224, hd_num=6):
    def _padding_224(frames):
        _, _, H, W = frames.shape
        tar = int(np.ceil(H / 224) * 224)
        top_padding = (tar - H) // 2
        bottom_padding = tar - H - top_padding
        left_padding = 0
        right_padding = 0

        padded_frames = F.pad(
            frames,
            pad=[left_padding, right_padding, top_padding, bottom_padding],
            mode='constant', value=255
        )
        return padded_frames

    _, _, H, W = frames.shape
    trans = False
    if W < H:
        frames = frames.flip(-2, -1)
        trans = True
        width, height = H, W
    else:
        width, height = W, H

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)

    resized_frames = F.interpolate(
        frames, size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )
    padded_frames = _padding_224(resized_frames)

    if trans:
        padded_frames = padded_frames.flip(-2, -1)

    return padded_frames


def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(2,1)):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    # calculate the existing video aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the frames
    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame

def load_video_internVL(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list



def load_video_llava(video_path, max_frames_num, start_time=0, end_time=None):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))

    fps = vr.get_avg_fps() 
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else len(vr)
    
    start_frame = max(0, start_frame)
    end_frame = min(len(vr), end_frame)

    uniform_sampled_frames = np.linspace(start_frame, end_frame - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()

    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def load_video_internvideo(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    # print(frames.shape)
    T_, C, H, W = frames.shape

    sub_img = frames.reshape(
        1, T_, 3, H//resolution, resolution, W//resolution, resolution
    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

    glb_img = F.interpolate(
        frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
    ).to(sub_img.dtype).unsqueeze(0)

    frames = torch.cat([sub_img, glb_img]).unsqueeze(0)

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
    

def preprocess(model_path,tokenizer,video_path,image_processor):
    if 'internvl' in model_path.lower():
        pixel_values, num_patches_list = load_video_internVL(video_path, num_segments=32, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + 'We detected a crime act between people or people and animals in this video. Try to find it and describe the action.'
        return {
            "pixel_values": pixel_values,
            "question": question,
            "num_patches_list": num_patches_list
        }

    elif 'llava' in model_path.lower():
        video_frames = load_video_llava(video_path, 32,)# start_time=video_meta['start_time'], end_time=video_meta['end_time'])
        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\nWe detected a crime act of abuse between people or people and animals in this video. Try to find it and describe the action."

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
        video_tensor = load_video_internvideo(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6)
        video_tensor = video_tensor
        return {"video_tensor":video_tensor}

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
                        "fps": 1.0,
                    },
                    {"type": "text",
                     "text": "Describe this video."},
                ],
            }
            ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
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
        from qwen_vl_utils import process_vision_info
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "/home/share/chenhaoran/model_zoo/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
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
                                            instruction= "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
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
        return output_text






def main():
    import warnings

    # 屏蔽所有 UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    model_paths = [
        
        #"/home/share/chenhaoran/model_zoo/lmms-lab--llava-onevision-qwen2-7b-ov", # work
        #"/home/share/chenhaoran/model_zoo/OpenGVLab--InternVideo2_Chat_8B_InternLM2_5",# StaticCache / dimension mismatch
        #"/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-1B",# work
        #"/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-8B",# work
        #"/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-26B",# work
        #"/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-40B",# work
        #"/home/share/chenhaoran/model_zoo/lmms-lab--llava-onevision-qwen2-72b-ov", # cpu offload problem
        "/home/share/chenhaoran/model_zoo/Qwen2-VL-7B-Instruct" # work
        # qwen2-72b/2b
    ]

    #model_path = model_paths[0]
    for model_path in model_paths:
        print('loading model:',model_path)
        try:
            model,image_processor = get_model(model_path)
        
        except Exception as e:
            print(f"Error loading model:({model_path.split('/')[-1]}) ",e)
            continue


        video_path = "/home/share/dataset/OpenDataLab___UCF-Crime/raw/UCF-Crime/Anomaly-Detection-Dataset/Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4"

        tokenizer = None
        if 'qwen2-vl' not in model_path:
            print('loading tokenizer')
            tokenizer =  AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,use_fast=False)

        with torch.no_grad():
            #print(model_path,tokenizer,video_path,image_processor)
            inputs = preprocess(model_path,tokenizer,video_path,image_processor)
            ans = model_generate(model_path,model,inputs,tokenizer)
        print("answer:\n",model_path,'\n',ans)

        del model
        torch.cuda.empty_cache()
        
    
main()