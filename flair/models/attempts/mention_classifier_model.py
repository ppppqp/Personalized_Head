import logging
from pathlib import Path
from typing import List, Union, Optional, Dict

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Label, Span
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path, unzip_file
from flair.training_utils import Metric, Result, store_embeddings

log = logging.getLogger("flair")


class Memory():
    def __init__(self, tag_dictionary):
        self.entity_memory = {}
        self.entity_counts = {}
        self.tag_dictionary = tag_dictionary

    def add_knowledge(self, entity, vector):

        if entity not in self.entity_memory:
            self.entity_memory[entity] = vector
            self.entity_counts[entity] = 1
        else:
            self.entity_memory[entity] += vector
            self.entity_counts[entity] += 1

    def get_knowledge(self, entity) -> torch.tensor:

        if entity not in self.entity_memory:
            return torch.tensor([-1.] * (len(self.tag_dictionary) + 1), device=flair.device)

        vector = self.entity_memory[entity] / self.entity_counts[entity]

        confidence = torch.tensor([min(self.entity_counts[entity] / 10, 1.)])
        # print(confidence)

        tensor = torch.cat([vector, confidence], dim=0)
        tensor = tensor.to(flair.device)
        return tensor


class Knowledge():
    def __init__(self, tag_dictionary):
        self.entity_knowledge = {}
        self.tag_dictionary = tag_dictionary

    def add_knowledge(self, entity, tag):

        if entity not in self.entity_knowledge:
            label_counts = torch.zeros(len(self.tag_dictionary))
        else:
            label_counts = self.entity_knowledge[entity]

        idx = self.tag_dictionary.get_idx_for_item(tag)
        label_counts[idx] = label_counts[idx] + 1

        self.entity_knowledge[entity] = label_counts

    def get_knowledge(self, entity) -> torch.tensor:

        if entity not in self.entity_knowledge:
            return torch.tensor([-1] * len(self.tag_dictionary))

        return self.entity_knowledge[entity]


class MentionTagger(flair.nn.Model):
    def __init__(
        self,
        hidden_size: int,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        use_memory: bool = True,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        reproject_embeddings: Union[bool,int] = True,
        train_initial_hidden_state: bool = False,
        rnn_type: str = "LSTM",
        pickle_module: str = "pickle",
        beta: float = 1.0,
        loss_weights: Dict[str, float] = None,
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        If you set this to an integer, you can control the dimensionality of the reprojection layer
        :param locked_dropout: locked dropout probability
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for classes (tags) for the loss function
        (if any tag's weight is unspecified it will default to 1.0)

        """

        super(MentionTagger, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.use_memory: bool = use_memory
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.knowledge = Memory(tag_dictionary) if self.use_memory else None

        extra_word_states = len(tag_dictionary) + 1 if self.use_memory else 0

        self.embeddings = embeddings

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary

        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        self.beta = beta

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        self.pickle_module = pickle_module

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        embedding_dim: int = self.embeddings.embedding_length
        rnn_input_dim: int = embedding_dim

        # optional reprojection layer on top of word embeddings
        self.reproject_embeddings = reproject_embeddings
        if self.reproject_embeddings:
            if type(self.reproject_embeddings) == int:
                rnn_input_dim = self.reproject_embeddings

            self.embedding2nn = torch.nn.Linear(embedding_dim, rnn_input_dim)

        self.train_initial_hidden_state = train_initial_hidden_state
        self.bidirectional = True
        self.rnn_type = rnn_type

        # bidirectional LSTM on top of embedding layer
        if self.use_rnn:
            num_directions = 2 if self.bidirectional else 1

            if self.rnn_type in ["LSTM", "GRU"]:

                self.rnn = getattr(torch.nn, self.rnn_type)(
                    rnn_input_dim,
                    hidden_size,
                    num_layers=self.nlayers,
                    dropout=0.0 if self.nlayers == 1 else 0.5,
                    bidirectional=True,
                    batch_first=True,
                )
                # Create initial hidden state and initialize it
                if self.train_initial_hidden_state:
                    self.hs_initializer = torch.nn.init.xavier_normal_

                    self.lstm_init_h = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    self.lstm_init_c = Parameter(
                        torch.randn(self.nlayers * num_directions, self.hidden_size),
                        requires_grad=True,
                    )

                    # TODO: Decide how to initialize the hidden state variables
                    # self.hs_initializer(self.lstm_init_h)
                    # self.hs_initializer(self.lstm_init_c)

            # final linear map to tag space
            self.linear = torch.nn.Linear(
                hidden_size * num_directions + extra_word_states, len(tag_dictionary)
            )
        else:
            self.linear = torch.nn.Linear(
                self.embeddings.embedding_length + extra_word_states, len(tag_dictionary)
            )

        self.to(flair.device)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "train_initial_hidden_state": self.train_initial_hidden_state,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "use_memory": self.use_memory,
            "use_rnn": self.use_rnn,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "beta": self.beta,
            "weight_dict": self.weight_dict,
            "reproject_embeddings": self.reproject_embeddings,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if "use_locked_dropout" not in state.keys()
            else state["use_locked_dropout"]
        )
        train_initial_hidden_state = (
            False
            if "train_initial_hidden_state" not in state.keys()
            else state["train_initial_hidden_state"]
        )
        beta = 1.0 if "beta" not in state.keys() else state["beta"]
        weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
        reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
        if "reproject_to" in state.keys():
            reproject_embeddings = state["reproject_to"]

        model = MentionTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
            beta=beta,
            loss_weights=weights,
            reproject_embeddings=reproject_embeddings,
        )
        model.load_state_dict(state["state_dict"])
        return model

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size=32,
        all_tag_prob: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss = False,
        embedding_storage_mode="none",
    ):
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a List of Sentence
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name == None:
            label_name = self.tag_type

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence):
                sentences = [sentences]

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            dataloader = DataLoader(
                dataset=SentenceDataset(reordered_sentences), batch_size=mini_batch_size
            )

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            overall_loss = 0
            batch_no = 0
            for batch in dataloader:

                batch_no += 1

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {batch_no}")

                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                loss, logits = self.forward_loss(batch, also_return_logits=True)
                # print(logits)

                softmax_batch = F.softmax(logits, dim=1).cpu()
                # print(softmax_batch)
                scores_batch, prediction_batch = torch.max(softmax_batch, dim=1)

                # print(prediction_batch)

                i = 0
                for sentence in batch:

                    for token in sentence:
                        token.set_label(label_name, 'O')

                    for span in sentence.get_spans('ner'):

                        prefix = 'B-'
                        for token in span:
                            token.set_label(label_name, prefix + self.tag_dictionary.get_item_for_index(prediction_batch[i]))
                            prefix = 'I-'

                        i += 1
                        # print(span)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def _requires_span_F1_evaluation(self) -> bool:
        return True

    def _evaluate_with_span_F1(self, data_loader, embedding_storage_mode, mini_batch_size, out_path):
        eval_loss = 0

        batch_no: int = 0

        metric = Metric("Evaluation", beta=self.beta)

        lines: List[str] = []

        y_true = []
        y_pred = []

        for batch in data_loader:

            # predict for batch
            loss = self.predict(batch,
                                embedding_storage_mode=embedding_storage_mode,
                                mini_batch_size=mini_batch_size,
                                label_name='predicted',
                                return_loss=True)
            eval_loss += loss
            batch_no += 1

            for sentence in batch:

                # make list of gold tags
                gold_spans = sentence.get_spans(self.tag_type)
                gold_tags = [(span.tag, repr(span)) for span in gold_spans]
                # print(gold_tags)

                # make list of predicted tags
                predicted_spans = sentence.get_spans("predicted")
                predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]
                # print(predicted_tags)

                # check for true positives, false positives and false negatives
                for tag, prediction in predicted_tags:
                    if (tag, prediction) in gold_tags:
                        metric.add_tp(tag)
                    else:
                        metric.add_fp(tag)

                for tag, gold in gold_tags:
                    if (tag, gold) not in predicted_tags:
                        metric.add_fn(tag)

                tags_gold = []
                tags_pred = []

                # also write to file in BIO format to use old conlleval script
                if out_path:
                    for token in sentence:
                        # check if in gold spans
                        gold_tag = 'O'
                        for span in gold_spans:
                            if token in span:
                                gold_tag = 'B-' + span.tag if token == span[0] else 'I-' + span.tag
                        tags_gold.append(gold_tag)

                        predicted_tag = 'O'
                        # check if in predicted spans
                        for span in predicted_spans:
                            if token in span:
                                predicted_tag = 'B-' + span.tag if token == span[0] else 'I-' + span.tag
                        tags_pred.append(predicted_tag)

                        lines.append(f'{token.text} {gold_tag} {predicted_tag}\n')
                    lines.append('\n')

                y_true.append(tags_gold)
                y_pred.append(tags_pred)

        if out_path:
            with open(Path(out_path), "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        eval_loss /= batch_no

        detailed_result = (
            "\nResults:"
            f"\n- F1-score (micro) {metric.micro_avg_f_score():.4f}"
            f"\n- F1-score (macro) {metric.macro_avg_f_score():.4f}"
            '\n\nBy class:'
        )

        for class_name in metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                f"fn: {metric.get_fn(class_name)} - precision: "
                f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                f"f1-score: "
                f"{metric.f_score(class_name):.4f}"
            )

        result = Result(
            main_score=metric.micro_avg_f_score(),
            log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{metric.micro_avg_f_score():.4f}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        return result, eval_loss

    def evaluate(
        self,
        sentences: Union[List[Sentence], Dataset],
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: int = 8,
    ) -> (Result, float):

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # if span F1 needs to be used, use separate eval method
        return self._evaluate_with_span_F1(data_loader, embedding_storage_mode, mini_batch_size, out_path)

    def forward_loss(self,
                     sentences: Union[List[Sentence], Sentence],
                     sort=True,
                     also_return_logits = False) -> torch.tensor:

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                sentence_tensor, lengths, enforce_sorted=False, batch_first=True
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     sentence_tensor = self.word_dropout(sentence_tensor)
            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)

        # print(sentence_tensor.size())

        tag_list: List = []
        all_span_embeddings = []
        for s_id, sentence in enumerate(sentences):

            spans = sentence.get_spans('ner')

            for span in spans:

                # tag = torch.tensor(self.tag_dictionary.get_idx_for_item(span.tag), device=flair.device)
                # tag_list.append(tag)
                tag_list.append(self.tag_dictionary.get_idx_for_item(span.tag))

                word_embeddings = []

                for token in span:
                    # print(sentence_tensor[s_id, token.idx-1])
                    word_embeddings.append(sentence_tensor[s_id, token.idx-1].unsqueeze(0))

                word_embeddings = torch.cat(word_embeddings, dim=0).to(flair.device)
                # print(word_embeddings)
                pooled = torch.mean(word_embeddings, 0)

                if self.use_memory:
                    pooled = torch.cat([pooled, self.knowledge.get_knowledge(span.text)])

                all_span_embeddings.append(pooled)

        all_span_embeddings = torch.stack(all_span_embeddings, dim=0).to(flair.device)

        logits = self.linear(all_span_embeddings)
        tag_tensor = torch.tensor(tag_list, device=flair.device)

        loss = torch.nn.functional.cross_entropy(
            logits, tag_tensor, weight=self.loss_weights
        )

        if self.use_memory:
            softmax_batch = F.softmax(logits, dim=1).cpu()
            i = 0
            for sentence in sentences:
                for span in sentence.get_spans('ner'):
                    self.knowledge.add_knowledge(span.text, softmax_batch[i].detach().clone())
                    i += 1

        # print(self.knowledge.get_knowledge('New York'))

        if also_return_logits:
            return loss, logits

        return loss

    @staticmethod
    def _softmax(x, axis):
        # reduce raw values to avoid NaN during exp
        x_norm = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x_norm)
        return y / y.sum(axis=axis, keepdims=True)

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    @staticmethod
    def _filter_empty_string(texts: List[str]) -> List[str]:
        filtered_texts = [text for text in texts if text]
        if len(texts) != len(filtered_texts):
            log.warning(
                f"Ignore {len(texts) - len(filtered_texts)} string(s) with no tokens."
            )
        return filtered_texts

    @staticmethod
    def _fetch_model(model_name) -> str:

        model_map = {}

        aws_resource_path_v04 = "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4"
        hu_path: str = "https://nlp.informatik.hu-berlin.de/resources/models"

        model_map["ner"] = "/".join(
            [aws_resource_path_v04, "NER-conll03-english", "en-ner-conll03-v0.4.pt"]
        )

        model_map["ner-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "NER-conll03--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward-fast%2Bnews-backward-fast-normal-locked0.5-word0.05--release_4",
                "en-ner-fast-conll03-v0.4.pt",
            ]
        )

        model_map["ner-ontonotes"] = "/".join(
            [
                aws_resource_path_v04,
                "release-ner-ontonotes-0",
                "en-ner-ontonotes-v0.4.pt",
            ]
        )

        model_map["ner-ontonotes-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-ner-ontonotes-fast-0",
                "en-ner-ontonotes-fast-v0.4.pt",
            ]
        )

        for key in ["ner-multi", "multi-ner"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "release-quadner-512-l2-multi-embed",
                    "quadner-large.pt",
                ]
            )

        for key in ["ner-multi-fast", "multi-ner-fast"]:
            model_map[key] = "/".join(
                [aws_resource_path_v04, "NER-multi-fast", "ner-multi-fast.pt"]
            )

        for key in ["ner-multi-fast-learn", "multi-ner-fast-learn"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "NER-multi-fast-evolve",
                    "ner-multi-fast-learn.pt",
                ]
            )

        model_map["upos"] = "/".join(
            [
                aws_resource_path_v04,
                "POS-ontonotes--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
                "en-pos-ontonotes-v0.4.pt",
            ]
        )

        model_map["pos"] = "/".join(
            [
                hu_path,
                "release-pos-0",
                "en-pos-ontonotes-v0.5.pt",
            ]
        )

        model_map["upos-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-pos-fast-0",
                "en-pos-ontonotes-fast-v0.4.pt",
            ]
        )

        model_map["pos-fast"] = "/".join(
            [
                hu_path,
                "release-pos-fast-0",
                "en-pos-ontonotes-fast-v0.5.pt",
            ]
        )

        for key in ["pos-multi", "multi-pos"]:
            model_map[key] = "/".join(
                [
                    aws_resource_path_v04,
                    "release-dodekapos-512-l2-multi",
                    "pos-multi-v0.1.pt",
                ]
            )

        for key in ["pos-multi-fast", "multi-pos-fast"]:
            model_map[key] = "/".join(
                [aws_resource_path_v04, "UPOS-multi-fast", "pos-multi-fast.pt"]
            )

        model_map["frame"] = "/".join(
            [aws_resource_path_v04, "release-frame-1", "en-frame-ontonotes-v0.4.pt"]
        )

        model_map["frame-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-frame-fast-0",
                "en-frame-ontonotes-fast-v0.4.pt",
            ]
        )

        model_map["chunk"] = "/".join(
            [
                aws_resource_path_v04,
                "NP-conll2000--h256-l1-b32-p3-0.5-%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0",
                "en-chunk-conll2000-v0.4.pt",
            ]
        )

        model_map["chunk-fast"] = "/".join(
            [
                aws_resource_path_v04,
                "release-chunk-fast-0",
                "en-chunk-conll2000-fast-v0.4.pt",
            ]
        )

        model_map["da-pos"] = "/".join(
            [aws_resource_path_v04, "POS-danish", "da-pos-v0.1.pt"]
        )

        model_map["da-ner"] = "/".join(
            [aws_resource_path_v04, "NER-danish", "da-ner-v0.1.pt"]
        )

        model_map["de-pos"] = "/".join(
            [hu_path, "release-de-pos-0", "de-pos-ud-hdt-v0.5.pt"]
        )

        model_map["de-pos-tweets"] = "/".join(
            [
                aws_resource_path_v04,
                "POS-fine-grained-german-tweets",
                "de-pos-twitter-v0.1.pt",
            ]
        )

        model_map["de-ner"] = "/".join(
            [aws_resource_path_v04, "release-de-ner-0", "de-ner-conll03-v0.4.pt"]
        )

        model_map["de-ner-germeval"] = "/".join(
            [aws_resource_path_v04, "NER-germeval", "de-ner-germeval-0.4.1.pt"]
        )

        model_map["fr-ner"] = "/".join(
            [aws_resource_path_v04, "release-fr-ner-0", "fr-ner-wikiner-0.4.pt"]
        )
        model_map["nl-ner"] = "/".join(
            [hu_path, "dutch-ner_0", "nl-ner-bert-conll02-v0.5b.pt"]
        )
        model_map["nl-ner-rnn"] = "/".join(
            [hu_path, "dutch-ner-flair-0", "nl-ner-conll02-v0.5.pt"]
        )
        model_map["ml-pos"] = "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/malayalam-xpos-model.pt"
        model_map["ml-upos"] = "https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/malayalam-upos-model.pt"

        model_map["keyphrase"] = "/".join(
            [hu_path, "keyphrase-semeval2017-scibert", "keyphrase-en-scibert.pt"]
        )

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        # the historical German taggers by the @redewiegergabe project
        if model_name == "de-historic-indirect":
            model_file = Path(flair.cache_root)  / cache_dir / 'indirect' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/indirect.zip', cache_dir=cache_dir)
                unzip_file(Path(flair.cache_root)  / cache_dir / 'indirect.zip', Path(flair.cache_root)  / cache_dir)
            model_name = str(Path(flair.cache_root)  / cache_dir / 'indirect' / 'final-model.pt')

        if model_name == "de-historic-direct":
            model_file = Path(flair.cache_root)  / cache_dir / 'direct' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/direct.zip', cache_dir=cache_dir)
                unzip_file(Path(flair.cache_root)  / cache_dir / 'direct.zip', Path(flair.cache_root)  / cache_dir)
            model_name = str(Path(flair.cache_root)  / cache_dir / 'direct' / 'final-model.pt')

        if model_name == "de-historic-reported":
            model_file = Path(flair.cache_root)  / cache_dir / 'reported' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/reported.zip', cache_dir=cache_dir)
                unzip_file(Path(flair.cache_root)  / cache_dir / 'reported.zip', Path(flair.cache_root)  / cache_dir)
            model_name = str(Path(flair.cache_root)  / cache_dir / 'reported' / 'final-model.pt')

        if model_name == "de-historic-free-indirect":
            model_file = Path(flair.cache_root)  / cache_dir / 'freeIndirect' / 'final-model.pt'
            if not model_file.exists():
                cached_path('http://www.redewiedergabe.de/models/freeIndirect.zip', cache_dir=cache_dir)
                unzip_file(Path(flair.cache_root)  / cache_dir / 'freeIndirect.zip', Path(flair.cache_root)  / cache_dir)
            model_name = str(Path(flair.cache_root)  / cache_dir / 'freeIndirect' / 'final-model.pt')

        return model_name


    def __str__(self):
        return super(flair.nn.Model, self).__str__().rstrip(')') + \
               f'  (beta): {self.beta}\n' + \
               f'  (weights): {self.weight_dict}\n' + \
               f'  (weight_tensor) {self.loss_weights}\n)'


    def train(self, mode=True):
        super().train(mode=mode)

        if mode:
            self.knowledge = Memory(self.tag_dictionary)
            print(
                f"Train mode - resetting memory."
            )
        else:
            # memory is wiped each time we do evaluation
            print(f"Prediction mode - growing memory")

        print(self.knowledge.entity_counts)