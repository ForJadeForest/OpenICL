"""Direct Generation Inferencer"""

from typing import List, Optional

import more_itertools
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import PretrainedConfig

from openicl import PromptTemplate
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.icl_retriever import *
from openicl.utils.icl_common_utils import get_generation_prompt_list_from_retriever_indices, \
    get_generation_vision_x_from_retriever_indices
from openicl.utils.logging import get_logger

logger = get_logger(__name__)


class FlamingoGenInferencerOutputHandler(GenInferencerOutputHandler):
    origin_prompt_dict = {}
    output_dict = {}
    prediction_dict = {}
    results_dict = {}
    origin_image_dict = {}

    def __init__(self,
                 num: int,
                 accelerator: Optional[Accelerator] = None):
        super().__init__(num, accelerator)
        self.origin_image_dict = {}

    def save_orgin_image(self, image_list: List[str]):
        for idx, origin_prompt in enumerate(image_list):
            if self.accelerator is not None:
                idx = idx * self.accelerator.num_processes + self.accelerator.process_index
            self.origin_image_dict[str(idx)] = origin_prompt


class FlamingoGenInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.

    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class.
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file.
        api_name (:obj:`str`, optional): Name of API service.
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method.
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 image_processor=None,
                 image_field='',
                 autocast_context=None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs={"max_new_tokens": 100},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs
        self.image_processor = image_processor
        self.image_field = image_field
        self.autocast_context = autocast_context

    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, force_words=None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = FlamingoGenInferencerOutputHandler(num, self.accelerator)
        index = 0

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        
        # 3. Generate prompts for testing input
        logger.info('begin concat the prompt...')
        prompt_list = get_generation_prompt_list_from_retriever_indices(ice_idx_list, retriever, self.tokenizer,
                                                                        self.gen_field_replace_token,
                                                                        max_model_token_num=self.max_model_token_num,
                                                                        ice_template=ice_template,
                                                                        prompt_template=prompt_template)
        output_handler.save_orgin_prompts(prompt_list)

        # 4. Inference for prompts in each batch
        logger.info("Starting inference process...")
        for idx_list in more_itertools.chunked(tqdm(range(len(prompt_list)), disable=not self.is_main_process),
                                               self.batch_size):
            text_entry = [prompt_list[i] for i in idx_list]
            sub_ice_idx_list = [ice_idx_list[idx] for idx in idx_list]
            vision_x = get_generation_vision_x_from_retriever_indices(sub_ice_idx_list, retriever,
                                                                      self.image_processor, self.image_field)
            # 5-1. Inference with local model
            with self.autocast_context:
                with torch.no_grad():
                    tokenized_data = self.tokenizer.batch_encode_plus(text_entry, padding=True, return_tensors='pt').to(
                        self.device)
                    prompt_len = int(tokenized_data.attention_mask.shape[1])
                    if force_words is not None:
                        force_words_ids = [
                            self.tokenizer(force_words).input_ids,
                        ]
                        outputs = self.model.generate(vision_x=vision_x,
                                                    lang_x=tokenized_data.input_ids,
                                                    attention_mask=tokenized_data.attention_mask,
                                                    force_words_ids=force_words_ids,
                                                    num_beams=10,
                                                    eos_token_id=self.tokenizer.eos_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    **self.generation_kwargs)
                    else:
                        outputs = self.model.generate(vision_x=vision_x,
                                                    lang_x=tokenized_data.input_ids,
                                                    attention_mask=tokenized_data.attention_mask,
                                                    eos_token_id=self.tokenizer.eos_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    **self.generation_kwargs)
                    outputs = outputs.tolist()
                    complete_output = self.tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
                    generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                                            skip_special_tokens=True)

            # 5-3. Save current output
            for prediction, output in zip(generated, complete_output):
                output_handler.save_prediction_and_output(prediction, output, index)
                index = index + 1

        # 6. Output
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return [sample['prediction'] for sample in output_handler.results_dict.values()]
