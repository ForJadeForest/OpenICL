'''Random Retriever'''

from typing import List, Optional

from accelerate import Accelerator
from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger

logger = get_logger(__name__)


class DirRetriever(BaseRetriever):
    """Dir In-context Learning Retriever Class
        Class of Dir Retriever.

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
        rtr_idx_list,
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
        self.rtr_idx_list = rtr_idx_list

    def retrieve(self):
        return self.rtr_idx_list
