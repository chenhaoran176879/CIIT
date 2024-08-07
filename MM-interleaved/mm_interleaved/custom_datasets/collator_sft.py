from typing import Any
import numpy as np

import torch

from .wds_utils import init_tokenizer


class MultiImageCollator:
    def __init__(
        self,
        tokenizer_path,
        mode="train",
        generation_kwargs=None,
        padding="longest",
        ignore_image_loss_idx=-1,
    ):
        """
        Designed for VIST Dataset
        """

        self.tokenizer = init_tokenizer(tokenizer_path)
        self.mode = mode
        self.generation_kwargs = generation_kwargs
        self.padding = padding
        self.ignore_image_loss_idx = ignore_image_loss_idx

    def set_mode(self, mode):
        self.mode = mode

    def __call__(self, data_list) -> Any:
        if self.mode == "train":
            return self._call_for_train(data_list)
        elif self.mode == "generate_texts":
            return self._call_for_generate_texts(data_list)
        elif self.mode == "generate_images":
            return self._call_for_generate_images(data_list)
        elif self.mode == "generate_both":
            raise NotImplementedError(
                f"Get {self.mode}, please specify the exact mode before calling it"
            )
        elif self.mode == "generate_segm":
            return self._call_for_generate_images(data_list)
        else:
            raise NotImplementedError(
                f"collate_mode {self.mode} is NOT supported by far"
            )

    def _call_for_generate_texts(self, data_list):
        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []
        meta = []
        text_inputs = []

        for data in data_list:
            meta.append(data["meta"])

            images_tensor = data["images_tensor"]
            if len(images_tensor) > 0:
                images_tensor = self._convert_images_tensor(images_tensor)
                if isinstance(images_tensor, tuple):
                    images_tensor, images_tensor_dec = images_tensor
                    images_tensors_dec_all += images_tensor_dec
                images_tensors_all += images_tensor
                num_image_per_seq.append(len(images_tensor))

            text_inputs.append(data["text"])

        self.tokenizer.padding_side = "left"
        text_tensor = self.tokenizer(
            text_inputs,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = None
        if len(images_tensors_all) > 0:
            images_tensors = torch.stack(images_tensors_all, dim=0)

        image_tensors_dec = None
        if len(images_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        if len(num_image_per_seq) > 0:
            num_image_per_seq = torch.tensor(
                num_image_per_seq, dtype=torch.long, device=images_tensors.device
            )
        else:
            num_image_per_seq = None

        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def _call_for_generate_images(self, data_list):
        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []
        meta = []
        text_inputs = []
        target_image_idxs = []

        for data in data_list:
            meta.append(data["meta"])

            images_tensor = data["images_tensor"]
            assert len(images_tensor) > 0

            images_tensor = self._convert_images_tensor(images_tensor)
            if isinstance(images_tensor, tuple):
                images_tensor, images_tensor_dec = images_tensor
                images_tensors_dec_all += images_tensor_dec
            images_tensors_all += images_tensor
            num_image_per_seq.append(len(images_tensor))
            target_image_idxs.append(sum(num_image_per_seq) - 1)

            text_inputs.append(data["text"])

        self.tokenizer.padding_side = "right"
        text_tensor = self.tokenizer(
            text_inputs,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        image_tensors_dec = None
        if len(images_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )
        target_image_idxs = torch.tensor(
            target_image_idxs, dtype=torch.long, device=images_tensors.device
        )

        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            meta=meta,
            target_image_idxs=target_image_idxs,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def _call_for_train(self, data_list):
        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []
        meta = []
        text_inputs = []
        text_input_tensors = []
        image_loss_mask_all = []
        dataset_name = ''

        for data in data_list:
            images_tensor = data["images_tensor"]
            assert len(images_tensor) > 0
            meta.append(data["meta"])
            dataset_name=data['dataset_name']
            images_tensor = self._convert_images_tensor(images_tensor)
            if isinstance(images_tensor, tuple):
                images_tensor, images_tensor_dec = images_tensor
                images_tensors_dec_all += images_tensor_dec
            images_tensors_all += images_tensor
            num_image_per_seq.append(len(images_tensor))
            if self.ignore_image_loss_idx > 0:
                image_loss_mask = [1.] * len(images_tensor)
                image_loss_mask[self.ignore_image_loss_idx] = 0.
                image_loss_mask_all.append(image_loss_mask)

            if data.get('text_tensor'):
                text_input_tensors.append(data['text_tensor'])
            elif 'text' in data:
                text_inputs.append(data["text"])


        if len(text_input_tensors) == len(data_list):
            text_ids_list = [tensor.input_ids for tensor in text_input_tensors]
            attn_mask_list = [tensor.attention_mask for tensor in text_input_tensors]

            # padding to longest
            max_length = max([ids.size(1) for ids in text_ids_list])
            padded_text_ids_list = []
            padded_attn_mask_list = []
            for ids, mask in zip(text_ids_list, attn_mask_list):
                padding_length = max_length - ids.size(1)
                padded_ids = torch.nn.functional.pad(ids, (0, padding_length), value=self.tokenizer.pad_token_id)
                padded_text_ids_list.append(padded_ids)
                
                
                padded_mask = torch.nn.functional.pad(mask, (0, padding_length), value=0)
                padded_attn_mask_list.append(padded_mask)
            try:
                text_ids = torch.cat(padded_text_ids_list, dim=0)
                attn_mask = torch.cat(padded_attn_mask_list, dim=0)
                #print("successfully cat:",text_ids.shape,attn_mask.shape)
            except Exception as e:
                torch.set_printoptions(threshold=torch.inf)
                raise ValueError("cannot cat torch tensor:",text_ids_list)


        
        elif len(text_inputs) == len(data_list):
            print("tokenize text in collator")
            self.tokenizer.padding_side = "right"
            text_tensor = self.tokenizer(
                text_inputs,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                padding=self.padding,
                return_tensors="pt",
                return_attention_mask=True,
            )
            text_ids = text_tensor.input_ids
            attn_mask = text_tensor.attention_mask
            print(f"{text_ids=},{attn_mask=}")

        
        #torch.set_printoptions(threshold=torch.inf)
        #print(f"features in collator:\n {data_list=}\n{text_ids=}\n{len(text_ids)=}\n")
        #torch.set_printoptions(profile="default")
        from time import sleep
        sleep(2)

        images_tensors = torch.stack(images_tensors_all, dim=0)
        image_tensors_dec = None
        if len(images_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        image_loss_mask = None
        if len(image_loss_mask_all) > 0:
            image_loss_mask = torch.tensor(
                image_loss_mask_all, device=images_tensors.device
            )

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        meta = dict(
            meta=meta,
            dataset_name=dataset_name
        )
        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            texts = text_inputs,
            text_ids=text_ids,
            attention_mask=attn_mask,
            meta=meta,
            image_loss_mask=image_loss_mask,
        )
        
        return data

    def _convert_images_tensor(self, images_tensor):
        if isinstance(images_tensor[0], tuple):
            images_tensor_dec = [i[1] for i in images_tensor]
            images_tensor = [i[0] for i in images_tensor]
            map_fn = (
                torch.from_numpy
                if isinstance(images_tensor[0], np.ndarray)
                else lambda x: x
            )
            images_tensor = [map_fn(image_tensor) for image_tensor in images_tensor]
            images_tensor_dec = [
                map_fn(image_tensor) for image_tensor in images_tensor_dec
            ]
            return images_tensor, images_tensor_dec
        else:
            map_fn = (
                torch.from_numpy
                if isinstance(images_tensor[0], np.ndarray)
                else lambda x: x
            )
            images_tensor = [map_fn(image_tensor) for image_tensor in images_tensor]
            return images_tensor
