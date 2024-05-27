import clip
import torch
import torch.distributed as dist
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from tqdm import tqdm
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


from CIITDataset import CIITDataset,is_img,is_text
from torch.utils.data import  DataLoader,DistributedSampler




class CLIPTower:
    def __init__(self, model_path, device=None):
        self.clip_model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    def forward(self, data):
        if is_img(data):
            images = self.processor(images=data,return_tensors='pt',padding=True)
            features = self.clip_model.get_image_features(**images)
        
        elif is_text(data):
            inputs = self.tokenizer(text=data,return_tensors='pt',padding=True,truncation=True)
            features = self.clip_model.get_text_features(**inputs)
        
        else: 
            raise TypeError
        
        return features

    @torch.no_grad()
    def calculate_clip_score(self,batch_1, batch_2):
        # default img batch_1, text batch_2
        logit_scale = self.clip_model.logit_scale.exp()
        features_1 = self.forward(batch_1)
        features_2 = self.forward(batch_2)
        features_1 = features_1 / features_1.norm(dim=1,keepdim=True).to(torch.float32)
        features_2 = features_2 / features_2.norm(dim=1,keepdim=True).to(torch.float32)
        score = logit_scale * torch.matmul(features_1 , features_2.T)
        return score

 
def dummy_save_clip_score(metainfo,score):
    
    pass


def main(args):
    model = CLIPTower(model_path=args.clip_path,device=args.device)
    dataset = CIITDataset()
    for idx in range(0,len(dataset)):
        data = dataset[idx]
        multimodal_context = data['multimodal_context']
        batch_1 = multimodal_context[::2]
        batch_2 = multimodal_context[1::2]
        score = model.calculate_clip_score(batch_1=batch_1,batch_2=batch_2)
        print(score)




if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size to use')
    parser.add_argument('--clip-path', type=str, default='/home/share/chenhaoran/model_zoo/clip-vit-large-patch14-336/',
                        help='CLIP model to use')
    parser.add_argument('--num-workers', type=int,default=1,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use. Like cuda, cuda:0 or cpu')
    args = parser.parse_args()
    
    #dist.init_process_group(backend='nccl')
    #local_rank = dist.get_rank()
    #torch.cuda.set_device(local_rank)
    #device = torch.device(f'cuda:{local_rank}')
    main(args)


