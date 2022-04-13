import logging
from pathlib import Path
from typing import List, Union, Callable, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, Label, Token, space_tokenizer, DataPoint
from flair.datasets import SentenceDataset, StringDataset
from flair.file_utils import cached_path
from flair.training_utils import (
    convert_labels_to_one_hot,
    Metric,
    Result,
    store_embeddings,
)

log = logging.getLogger("flair")


class DataClassifier(flair.nn.Model):

    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_dictionary: Dictionary,
    ):
        super(DataClassifier, self).__init__()

        self.document_embeddings: flair.embeddings.DocumentRNNEmbeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        self.label_type = label_dictionary.label_type

    def forward(self, sentences) -> List[List[float]]:

        self.document_embeddings.embed(sentences)

        text_embedding_list = [
            sentence.embedding.unsqueeze(0) for sentence in sentences
        ]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def _get_state_dict(self):
        pass

    @staticmethod
    def _init_model_with_state_dict(state):

        # model.load_state_dict(state["state_dict"])
        pass

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):

        pass
