"""Topk Retriever"""

import copy
from typing import Optional

import faiss
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from openicl import DatasetReader
from openicl.icl_dataset_reader import DatasetEncoder
from openicl.icl_retriever import BaseRetriever
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
from openicl.utils.logging import get_logger
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

logger = get_logger(__name__)


class MMTopkRetriever(BaseRetriever):
    """MultiModal In-context Learning Retriever Class
        Class of MultiModal Topk Retriever.

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
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
    """

    model = None

    def __init__(
        self,
        dataset_reader: DatasetReader,
        ice_separator: Optional[str] = '\n',
        ice_eos_token: Optional[str] = '\n',
        prompt_eos_token: Optional[str] = '',
        ice_num: Optional[int] = 1,
        index_split: Optional[str] = 'train',
        test_split: Optional[str] = 'test',
        clip_model_name: Optional[str] = 'openai/clip-vit-base-patch32',
        mode: Optional[str] = 'i2t',
        index_field: Optional[str] = 'text',
        test_field: Optional[str] = 'image',
        batch_size: Optional[int] = 1,
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
        self.clip_model_name = clip_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        logger.info(f'begin load {clip_model_name} text encodcer')
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            clip_model_name
        ).to(self.device)
        logger.info(f'begin load {clip_model_name} image encodcer')
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            clip_model_name
        ).to(self.device)

        self.text_encoder.eval()
        self.vision_encoder.eval()

        logger.info(f'begin load {clip_model_name} processor and tokenizer...')
        self.img_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.tokenzier = AutoTokenizer.from_pretrained(clip_model_name)

        encoding_method_map = {'i': self.encode_img, 't': self.encode_text}
        index_encoding = mode.split('2')[1]
        test_encoding = mode.split('2')[0]

        self.index_features = encoding_method_map[index_encoding](
            self.index_ds, index_field
        )
        self.test_features = encoding_method_map[test_encoding](
            self.test_ds, test_field
        )
        emb_dim = self.index_features.shape[1]
        self.index = faiss.IndexFlatIP(emb_dim)
        
        logger.info(f'begin add the index for emb dim: {self.index_features.shape}')
        self.index.add(self.index_features)

    @torch.inference_mode()
    def encode_text(self, ds, text_field):
        logger.info(f'now begin tokenizer field: {text_field}')
        remove_columns = ds.column_names

        text_ds = ds.map(
            lambda x: self.tokenzier(x[text_field], padding=True, return_tensors='pt'),
            batched=True,
            batch_size=self.batch_size,
            remove_columns=remove_columns,
        )
        text_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dataloader = DataLoader(text_ds, batch_size=self.batch_size, shuffle=False)
        logger.info(
            f'use {self.clip_model_name} to encode the text field: {text_field}'
        )
        bar = tqdm.tqdm(dataloader, disable=not self.is_main_process)

        feature_list = []
        for batch_data in bar:
            features = self.text_encoder(
                input_ids=batch_data['input_ids'].to(self.device),
                attention_mask=batch_data['attention_mask'].to(self.device),
            ).text_embeds
            features /= features.norm(dim=-1, keepdim=True)
            feature_list.append(features)
        features = torch.cat(feature_list, dim=0)
        return features.cpu().detach().numpy()

    @torch.inference_mode()
    def encode_img(self, ds, img_field):
        logger.info(f'now begin processor img field: {img_field}')
        remove_columns = ds.column_names
        if isinstance(ds[0][img_field], str):
            logger.info(f'detect the img_field is str, now begin read the path')
            img_ds = ds.map(
                lambda x: self.img_processor(
                    images=Image.open(x[img_field]).convert('RGB'), return_tensors="pt"
                ),
                remove_columns=remove_columns,
            )
        elif isinstance(ds[0][img_field], Image.Image):
            img_ds = ds.map(
                lambda x: self.img_processor(images=x[img_field], return_tensors="pt"),
                remove_columns=remove_columns,
            )
        else:
            raise ValueError(
                f'the data in image field should be str or PIL.Image.Image, but got {type(ds[0][img_field])}'
            )
        img_ds.set_format(type="torch", columns=["pixel_values"])
        dataloader = DataLoader(img_ds, batch_size=self.batch_size, shuffle=False)
        logger.info(f'use {self.clip_model_name} to encode the img field: {img_field}')
        bar = tqdm.tqdm(dataloader, disable=not self.is_main_process)

        feature_list = []
        for batch_data in bar:
            features = self.vision_encoder(
                batch_data['pixel_values'].squeeze(dim=1).to(self.device)
            ).image_embeds
            features /= features.norm(dim=-1, keepdim=True)
            feature_list.append(features)
        features = torch.cat(feature_list, dim=0)
        return features.cpu().detach().numpy()

    def retrieve(self):
        idx_list = self.index.search(self.test_features, self.ice_num)[1].tolist()
        return idx_list
