import copy
import json
import logging
import os
import pathlib
import warnings
import re
from collections import defaultdict

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from .wds_utils import init_tokenizer
import torch
import transformers
from torch.utils.data import Dataset
from .loader import BaseDataset
from PIL import Image

def is_img(data):
    if isinstance(data,list):
        return all(isinstance(elem,Image.Image) for elem in data)
    return isinstance(data,Image.Image)

def is_text(data):
    if isinstance(data,list):
        return all(isinstance(elem,str) for elem in data)
    return isinstance(data,str)

baidu_file = "/home/yidongyi/ImageDataSets/dingchenglin_imgs/baidujingyan_tuwen_data_2023-12-06.jsonl"
toutiao_file = "/home/yidongyi/ImageDataSets/dingchenglin_imgs/tuwen_2023-11-02.jsonl"
   
'''
data_example=
    {"title": "画图3D如何制作3D涂鸦字体", 
    "news_date": "2020-12-03 11:36", 
    "platform": "toutiao_tuwen", 
    "content": "我们在用画图3D制作字体的时候，我们可以使用3D涂鸦效果来制作立体字体，下面几步教会你！\n
    工具/原料\n电脑系统：windows10 家庭版\n系统类别：64位操作系统\n方法/步骤\n1.打开带有win10系统的电脑，打开程序，点击画图3D；
    如图\n<image>e2aefe781431dfb657ff19d012cf02532e636329</image>\n
    2.点击新建文件，点击裁剪，选择一个你想要的尺寸；如图\n<image>54a89daee8d7592a827caece9f31dfb6336c6729</image>\n
    3.点击3D形状，点击3D涂鸦，选择第一个画笔；如图\n<image>82eff6d7592ae3ef82eb5a8754b6326c56666429</image>\n
    4.更改涂鸦画笔的粗细，并选择一种你喜欢的颜色；如图\n<image>586bfdefe0781431bb257cdadc6699cf03536229</image>\n
    5.直接在画布上写上数字，可以调整锚点来更改角度；如图\n<image>e1390a31dfb6326c31f5d77989532f6322856029</image>\n
    6.再次把画笔的粗细调大，更改颜色，再次绘制；如图\n<image>332d496699cf025346f06d336b36e29146e85f29</image>\n
    注意事项\n如果对你有帮助请点赞关注", 
    "news_class": "",
      "news_url": "https://jingyan.baidu.com/article/200957619b26308a0621b41f.html", 
      "AIGC_info": [{"video_urls": [], "img_urls":
        ["https://exp-picture.cdn.bcebos.com/e2aefe781431dfb657ff19d012cf02532e636329.jpg",
        "https://exp-picture.cdn.bcebos.com/54a89daee8d7592a827caece9f31dfb6336c6729.jpg", 
        "https://exp-picture.cdn.bcebos.com/82eff6d7592ae3ef82eb5a8754b6326c56666429.jpg", 
      "https://exp-picture.cdn.bcebos.com/586bfdefe0781431bb257cdadc6699cf03536229.jpg",
        "https://exp-picture.cdn.bcebos.com/e1390a31dfb6326c31f5d77989532f6322856029.jpg", 
        "https://exp-picture.cdn.bcebos.com/332d496699cf025346f06d336b36e29146e85f29.jpg"], 
        "hash_codes": ["e2aefe781431dfb657ff19d012cf02532e636329", "54a89daee8d7592a827caece9f31dfb6336c6729",
          "82eff6d7592ae3ef82eb5a8754b6326c56666429", "586bfdefe0781431bb257cdadc6699cf03536229", "e1390a31dfb6326c31f5d77989532f6322856029", 
          "332d496699cf025346f06d336b36e29146e85f29"],
            "img_save_dir": "/home/yidongyi/ImageDataSets/dingchenglin_imgs/baidujingyan_tuwen/0"}],
            "create_time": "2023-11-22 10:14:14"}

''' 

def get_interleave_form(text):
    # 定义正则表达式模式来匹配<image>标记及链接部分
    pattern = r'<image>[^<]+</image>'

    # 使用findall()方法找到所有匹配的<image>标记及链接部分
    matches = re.findall(pattern, text)

    # 对于每个匹配项，提取<image>标记并用其前后的文本进行分割
    result = []
    last_end = 0
    for match in matches:
        start_index = text.find(match)
        if start_index > last_end:
            result.append(text[last_end:start_index])
        result.append(match)
        last_end = start_index + len(match)

    # 添加最后一个匹配项之后的文本
    if last_end < len(text):
        result.append(text[last_end:])

    return result


class CIITDataset(BaseDataset):
    def __init__(
        self, 
        annt_files,
        image_root=None,
        transforms=None,
        tokenizer_path=None,
        #collate_mode="generate_texts",
        num_img_token=32,
        add_eos="",
        add_soi_token=True,
        total_sample_num=None,

        ):

        super().__init__()
        self.annt_files = annt_files
        self.max_test_sample_num = total_sample_num if total_sample_num else 1e10

        #self.collate_mode = collate_mode
        self.img_root = image_root
        self.transforms = transforms
        self.add_soi_token = add_soi_token
        self.add_eos = add_eos
        self.num_img_token = num_img_token


        #assert collate_mode in ["train", "generate_texts", "generate_images"]
        
        self.image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            self.image_subseq = "<|beginofimage|>" + self.image_subseq


        self.tokenizer = init_tokenizer(tokenizer_path)
        self.load_database()
        print(f"length of the dataset is {self.__len__()}")

    def load_database(self):
        self.data = []
        annt_files = self.annt_files#[os.path.join(self.annt_root,x) for x in self.annt_files]
        for file_path in annt_files:
            with open(file_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
                    if len(self.data) > self.max_test_sample_num: return
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # returns texts in list and img in PIL.Image
        meta_info = self.data[idx]
        text = meta_info['content']
        #if meta_info.get('title',None):
        #    text = meta_info['title'] + '\n' + text
        


        text_interleaved = get_interleave_form(text)
        image_file = os.path.join(self.img_root, meta_info['AIGC_info'][0]['img_save_dir'])

        multimodal_context = []
        image_tensors = []
        multimodal_text = ""
        for modality in text_interleaved:
            if modality.startswith('<image>') and modality.endswith('</image>'):
                img_id = modality[len("<image>"):-len("</image>")]
                img_path = os.path.join(image_file,img_id+'.jpg')
                try:
                    image = Image.open(img_path).convert("RGB")
                    
                except Exception as e:
                    print(f'Value Error: {img_path} does not exist or cannot open.')
                    continue
                
                if self.transforms:
                    image = self.transforms(image)
                
                multimodal_context.append(image)
                multimodal_text += f"{self.image_subseq}"
                image_tensors.append(image)
            else:
                multimodal_context.append(modality)
                multimodal_text += modality

        if self.add_eos:
            text += self.add_eos
        
        return  {
                'text':multimodal_text,
                'images_tensor': image_tensors,
                #'multimodal_context':multimodal_context,
                'meta':meta_info
                  }


    def get_meta_info(self,idx):
        return self.data[idx]

