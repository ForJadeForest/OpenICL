"""PPL Inferencer"""

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import (
    BaseInferencer,
    PPLInferencerOutputHandler,
)
from openicl.utils.logging import get_logger
from openicl.utils.api_service import *
from typing import List, Union, Optional
from tqdm import tqdm
from tqdm import trange
from transformers import PretrainedConfig
from accelerate import Accelerator

logger = get_logger(__name__)


class ICVPPLInferencer(BaseInferencer):
    """PPL In-context Learning Inferencer Class
        Perplexity-based In-context Learning Inferencer.

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
        labels (:obj:`List`, optional): A list of labels for all classes.
    """

    def __init__(
        self,
        icv_encoder,
        icv_tokenizer,
        model_name: Optional[str] = "gpt2-xl",
        tokenizer_name: Optional[str] = None,
        max_model_token_num: Optional[int] = None,
        model_config: Optional[PretrainedConfig] = None,
        batch_size: Optional[int] = 1,
        accelerator: Optional[Accelerator] = None,
        output_json_filepath: Optional[str] = "./icl_inference_output",
        output_json_filename: Optional[str] = "predictions",
        api_name: Optional[str] = None,
        labels: Optional[List] = None,
        model_parallel: Optional[bool] = False,
        device="cuda",
        **kwargs,
    ) -> None:
        super().__init__(
            model_name,
            tokenizer_name,
            max_model_token_num,
            model_config,
            batch_size,
            accelerator,
            output_json_filepath,
            output_json_filename,
            api_name,
            model_parallel,
            device,
            **kwargs,
        )
        self.labels = labels
        self.icv_encoder = icv_encoder
        self.icv_tokenizer = icv_tokenizer

    def inference(
        self,
        retriever: BaseRetriever,
        ice_template: Optional[PromptTemplate] = None,
        prompt_template: Optional[PromptTemplate] = None,
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> List:
        # 1. Preparation for output logs
        output_handler = PPLInferencerOutputHandler(self.accelerator)

        sub_predictions = []
        ppl = []
        ice = []

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        shot_num = len(ice_idx_list[0])

        # 3. Get labels of all the classes
        if self.labels is None:
            labels = retriever.get_labels(
                ice_template=ice_template, prompt_template=prompt_template
            )
        else:
            labels = self.labels

        # 4. Generate in-context examples for testing inputs
        for idx in range(len(ice_idx_list)):
            ice.append(
                retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template)
            )
        output_handler.save_ice(ice)

        # 5. Calculating PPL for prompts in each label's class
        for label in labels:
            index = 0
            query_prompt_list = []
            sub_ppl_list = []

            # 5.1 Generate prompts of current label and truncate

            for idx in range(len(ice_idx_list)):
                query_prompt = retriever.generate_label_prompt(
                    idx,
                    ice="",
                    label=label,
                    ice_template=ice_template,
                    prompt_template=prompt_template,
                    remain_sep=False,
                )
                query_prompt_list.append(query_prompt)

            # 5.2 Get PPL
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(
                0,
                len(query_prompt_list),
                self.batch_size,
                disable=not self.is_main_process,
            ):
                sub_ice_list = ice[idx : idx + self.batch_size]
                sub_query_prompt = query_prompt_list[idx : idx + self.batch_size]

                with torch.no_grad():
                    if self.icv_tokenizer is not None:
                        inputs = self.icv_tokenizer(
                            sub_ice_list,
                            padding=True,
                            return_tensors="pt",
                            truncation=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        icv_outputs = self.icv_encoder(
                            inputs,
                            torch.tensor([shot_num for _ in range(self.batch_size)]),
                        )
                    else:
                        # It's single_icv_encoder
                        icv_outputs = self.icv_encoder(None, None)
                    sub_res = self.__get_ppl(
                        query_input=sub_query_prompt,
                        in_context_vector=icv_outputs.in_context_vector,
                        alpha=icv_outputs.alpha,
                    ).tolist()
                for res, prompt in zip(sub_res, sub_query_prompt):
                    sub_ppl_list.append(res)
                    output_handler.save_prompt_and_ppl(
                        label, prompt[len(ice[idx]) :], prompt, res, index
                    )
                    index = index + 1
            ppl.append(sub_ppl_list)

        # 6. Get lowest PPL class as predictions
        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(min(single_ppl))])
        output_handler.save_predictions(sub_predictions)

        # 7. Output
        output_handler.subprocess_write_to_json(
            output_json_filepath, output_json_filename
        )
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return [sample["prediction"] for sample in output_handler.results_dict.values()]

    def __get_ppl(
        self,
        query_input: List[str],
        in_context_vector,
        alpha,
        mask_length=None,
    ):

        self.tokenizer.padding_side = "right"

        query_input = self.tokenizer(
            query_input, padding=True, return_tensors="pt", truncation=True
        )
        query_input = {k: v.to(self.device) for k, v in query_input.items()}
        outputs = self.model(
            **query_input,
            in_context_vector=in_context_vector,
            alpha=alpha,
        )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = query_input["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.tokenizer.pad_token_id
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (
            (query_input["input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .cpu()
            .numpy()
        )
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
