import json
import os
import re
from collections import defaultdict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
baidu_file = "/home/yidongyi/ImageDataSets/dingchenglin_imgs/baidujingyan_tuwen_data_2023-12-06.jsonl"
toutiao_file = "/home/yidongyi/ImageDataSets/dingchenglin_imgs/tuwen_2023-11-02.jsonl"


class CIITDataset(Dataset):
    def __init__(self, file_path_list):
        self.data = []
        for file_path in file_path_list:
            with open(file_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



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

def get_interleave_distribution(dataset):
    lens_dict = defaultdict(int)
    for i in range(len(dataset)):
        lens_dict[len(get_interleave_form(dataset[i]['content']))]+=1

    with open("/mnt/lustre/chenhaoran/CIIT/dataset/interleave_summary.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(lens_dict))
    
    return lens_dict

def plot_interleave_distribution(data_dict,select_minimal_rate = 0.3):
    # 将字典的键转换为整数，并按键的大小排序
    sorted_keys = sorted(map(int, data_dict.keys()))
    
    n = int(select_minimal_rate*len(sorted_keys))
    sorted_keys=sorted_keys[:n]

    values = [data_dict[str(key)] for key in sorted_keys]
    
    # 绘制曲线图
    plt.plot(sorted_keys, values, marker='o')

    # 添加标题和标签
    plt.title('Interleave Distribution')
    plt.xlabel('interleave times')
    plt.ylabel('data num')

    # 显示图形
    plt.show()
    plt.savefig(f"/mnt/lustre/chenhaoran/CIIT/dataset/interleave_summary_{select_minimal_rate}.jpg")


def get_start_modality_distribution(dataset):
    start_modality_dict = defaultdict(int)

    pass

def scripts_save():
    with open("/mnt/lustre/chenhaoran/CIIT/dataset/interleave_summary.json",'r',encoding='utf-8') as f:
        lens_dict = json.load(f)

    for rate in range(1,11,1):
        plot_interleave_distribution(lens_dict,select_minimal_rate=float(rate)/10)

if __name__ == '__main__':
    dataset = CIITDataset([baidu_file,toutiao_file])
    print(len(dataset))