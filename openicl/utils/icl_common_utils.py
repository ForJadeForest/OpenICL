import torch
from torch.utils.data import DataLoader
from openicl.icl_retriever import BaseRetriever
from typing import List, Union, Optional
from openicl import PromptTemplate
from accelerate import Accelerator


def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
    dataloader = DataLoader(datalist, batch_size=batch_size)
    return dataloader


def get_generation_prompt_list_from_retriever_indices(ice_idx_list: List[List[int]], retriever: BaseRetriever,
                                                      tokenizer, gen_field_replace_token: str,
                                                      max_model_token_num: Optional[int] = None,
                                                      ice_template: Optional[PromptTemplate] = None,
                                                      prompt_template: Optional[PromptTemplate] = None):
    prompt_list = []
    for idx, ice_idx in enumerate(ice_idx_list):
        ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
        prompt = retriever.generate_prompt_for_generate_task(idx, ice, gen_field_replace_token=gen_field_replace_token,
                                                             ice_template=ice_template, prompt_template=prompt_template)
        if max_model_token_num is not None and tokenizer is not None:
            prompt_token_num = get_input_token_num(tokenizer, prompt)
            while len(ice_idx) > 0 and prompt_token_num > max_model_token_num:
                ice_idx = ice_idx[:-1]
                ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
                prompt = retriever.generate_prompt_for_generate_task(idx, ice,
                                                                     gen_field_replace_token=gen_field_replace_token,
                                                                     ice_template=ice_template,
                                                                     prompt_template=prompt_template)
                prompt_token_num = get_input_token_num(tokenizer, prompt)
        prompt_list.append(prompt)
    return prompt_list


def get_generation_vision_x_from_retriever_indices(ice_idx_list: List[List[str]], retriever: BaseRetriever,
                                                   image_processor, image_field='image'):
    image_inputs = []
    for idx, ice_idx, in enumerate(ice_idx_list):
        image = retriever.index_ds[idx][image_field]
        ice_image_list = [retriever.index_ds[i][image_field] for i in ice_idx]
        ice_image_list = [image_processor(img) for img in ice_image_list] + [image_processor(image)]
        ice_image = torch.stack(ice_image_list, dim=0)
        image_inputs.append(ice_image)

    image_inputs = torch.stack(image_inputs, dim=0)
    image_inputs = image_inputs.unsqueeze(dim=2)
    return image_inputs


def get_input_token_num(tokenizer, input):
    return len(tokenizer(input, verbose=False)['input_ids'])
