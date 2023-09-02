'''Random Retriever'''

from typing import List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from tqdm import trange
from transformers import AutoProcessor, AutoTokenizer

logger = get_logger(__name__)


class MMICLMRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class
        Class of Random Retriever.

    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        seed (`int`, optional): Seed for the random number generator.
    """

    def __init__(
        self,
        dataset_reader: DatasetReader,
        iclm_model: torch.nn.Module,
        processor,
        tokenzier,
        ice_separator: Optional[str] = '\n',
        ice_eos_token: Optional[str] = '\n',
        prompt_eos_token: Optional[str] = '',
        ice_num: Optional[int] = 1,
        index_split: Optional[str] = 'train',
        test_split: Optional[str] = 'test',
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        super().__init__(
            dataset_reader,
            ice_separator,
            ice_eos_token,
            prompt_eos_token,
            ice_num,
            index_split,
            test_split,
            accelerator,
        )
        self.model = iclm_model
        if isinstance(processor, str):
            self.processor = AutoProcessor.from_pretrained(processor)
        else:
            self.processor = processor

        if isinstance(tokenzier, str):
            self.tokenzier = AutoTokenizer(tokenzier)
        else:
            self.tokenizer = tokenzier

    @torch.inference_mode()
    def generation_ice_iclm(self, test_data, shot_num):
        input_ids = None
        device = next(self.model.parameters()).device
        self.model.eval()
        img_input = self.processor(images=test_data['image'], return_tensors='pt')

        for _ in range(shot_num):
            outputs = self.model(img_input=img_input, ice_input=input_ids)
            next_token_idx = torch.softmax(outputs.logits[:, -1, :], dim=-1).argmax(
                dim=-1
            )
            if input_ids is None:
                input_ids = torch.tensor(
                    [[next_token_idx]], device=device
                )  # 1, seq_len
            else:
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token_idx]], device=device)], dim=1
                )
        return input_ids.detach().cpu().tolist()[0]

    def retrieve(self):
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for i in trange(len(self.test_ds), disable=not self.is_main_process):
            test_data = self.test_ds[i]
            idx_list = self.generation_ice_iclm(test_data, self.ice_num)
            rtr_idx_list.append(idx_list)
        return rtr_idx_list
