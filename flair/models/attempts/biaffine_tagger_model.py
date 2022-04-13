import logging
from pathlib import Path
from typing import List, Union, Optional, Callable, Dict

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer, DataPoint
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.models import SequenceTagger
from flair.training_utils import Metric, Result, store_embeddings

log = logging.getLogger("flair")

class BiaffineAttention(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(BiaffineAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = torch.nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        print(self.U.size())
        torch.nn.init.xavier_uniform_(self.U)

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)


class BiaffineTagger(flair.nn.Model):
    def __init__(
        self,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        hidden_size: int = 256,
        biaffine_size: int = 200,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        reproject_embeddings: Union[bool,int] = True,
        rnn_type: str = "LSTM",
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        If you set this to an integer, you can control the dimensionality of the reprojection layer
        :param locked_dropout: locked dropout probability
        """

        super(BiaffineTagger, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_size = hidden_size
        self.biaffine_size = biaffine_size
        self.rnn_layers: int = rnn_layers

        self.trained_epochs: int = 0

        self.embeddings = embeddings

        # set the dictionaries
        self.tag_dictionary: Dictionary = tag_dictionary

        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # initialize dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

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

        self.rnn_type = rnn_type

        # bidirectional LSTM on top of embedding layer
        if self.rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(torch.nn, self.rnn_type)(
                rnn_input_dim,
                hidden_size,
                num_layers=self.nlayers,
                dropout=0.0 if self.nlayers == 1 else 0.5,
                bidirectional=True,
                batch_first=True,
            )
        rnn_output_size = 2 * hidden_size # RNN is bidirectional

        # maps to start and stop features
        self.ffnn_start = torch.nn.Linear(rnn_output_size, biaffine_size)
        self.ffnn_end = torch.nn.Linear(rnn_output_size, biaffine_size)

        # biaffine map
        self.biaffine_map = BiaffineAttention(biaffine_size, len(tag_dictionary))

        self.to(flair.device)

    # def forward_loss(
    #     self, data_points: Union[List[Sentence], Sentence]) -> torch.tensor:
    #     features = self.forward(data_points)
    #     return self._calculate_loss(features, data_points)

    def forward_loss(self, sentences: List[Sentence], also_return_logits = False):

        # embed sentences
        self.embeddings.embed(sentences)

        # make padded embedding tensor for all sentences in mini-batch (and return lengths of sentences)
        lengths, sentence_tensor = self._make_padded_embedding_tensor(sentences)

        # dropout
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        # optional reprojection layer
        if self.reproject_embeddings:
            sentence_tensor = self.embedding2nn(sentence_tensor)

        # RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )

        # if initial hidden state is trainable, use this state
        rnn_output, hidden = self.rnn(packed)

        # print(rnn_output.data.size())
        # asd
        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_output, batch_first=True
        )

        # more dropout after RNN
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        # map to tag space (create logits)
        start_features = self.ffnn_start(sentence_tensor)
        stop_features = self.ffnn_end(sentence_tensor)
        # print(start_features.size())
        # print(stop_features.size())

        result_matrix = self.biaffine_map(start_features, stop_features)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        matrix_size: int = max(lengths) * max(lengths)

        result_matrix = result_matrix.transpose(1, 3)
        # print(max(lengths))
        # print('---------------')
        # print(result_matrix)

        # asd
        # print(result_matrix.transpose(1, 3).size())

        # softmax_batch = F.softmax(result_matrix, dim=3)
        # print(softmax_batch.size())
        # softmax_batch = softmax_batch.view(1, 9, 5)
        # print(softmax_batch)
        # asd
        # predicted = torch.zeros(result_matrix.size(), dtype=torch.long, device=flair.device)
        gold = torch.zeros([result_matrix.size(0), result_matrix.size(1), result_matrix.size(2)], dtype=torch.long, device=flair.device)
        for s_id, sentence in enumerate(sentences):
            for span in sentence.get_spans('ner'):
                gold[s_id, span[0].idx-1, span[-1].idx-1] = self.tag_dictionary.get_idx_for_item(span.tag)
            # print(f'length sentence: {len(sentence)}')
            for i in range(len(sentence), max(lengths)):
                # print(i)
                for y in range(len(sentence), max(lengths)):
                    # print(y)
                    result_matrix[s_id, i, y] = 0.
                    # print(result_matrix[s_id, i, y])

        print(result_matrix.size())
        print(result_matrix)
        print(gold.size())
        print(gold)
        # asd

        # print(result_matrix)
        # result_matrix = result_matrix.squeeze(0)
        # print(predicted_labels.size())
        # print(result_matrix.view(1, 9, 5).size())

        # print(gold.size())
        # gold = gold.squeeze(0)
        # print(gold.size())
        # asd
        # print(gold.view(1, 9, 5).size())

        # print(result_matrix.contiguous().view(len(sentences) * matrix_size, 5))

        loss = torch.nn.functional.cross_entropy(
            result_matrix.contiguous().view(len(sentences) * matrix_size, 5),
            gold.contiguous().view(len(sentences) * matrix_size))
        #
        if also_return_logits:
            return result_matrix, loss
        #
        return loss

    def _calculate_loss(
        self, features: torch.tensor, sentences: List[Sentence]
    ) -> float:

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        for s_id, sentence in enumerate(sentences):
            # get the tags in this sentence
            tag_idx: List[int] = [
                self.tag_dictionary.get_idx_for_item(token.get_tag(self.tag_type).value)
                for token in sentence
            ]
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)

        score = 0
        for sentence_feats, sentence_tags, sentence_length in zip(
            features, tag_list, lengths
        ):
            sentence_feats = sentence_feats[:sentence_length]
            score += torch.nn.functional.cross_entropy(
                sentence_feats, sentence_tags
            )
        score /= len(features)
        return score

    def _obtain_labels(
        self,
        feature: torch.Tensor,
        batch_sentences: List[Sentence],
        label_name: str,
    ) -> List[List[Label]]:
        """
        Returns a list of to the most likely `Label` per token in each sentence.
        """

        # print(len(batch_sentences))

        # print(feature)
        softmaxed_matrix = F.softmax(feature, dim=3)
        # print(softmaxed_matrix)
        confidences, predictions = torch.max(softmaxed_matrix, dim=3)
        # print(predictions)
        # asd

        for s_no, sentence in enumerate(batch_sentences):
            for token in sentence:
                token.set_label(label_name, 'O')

            top_spans = []

            for start_no, start_token in enumerate(sentence):
                for stop_no, stop_token in enumerate(sentence):
                    if stop_no < start_no: continue

                    prediction = predictions[s_no, start_no, stop_no].item()
                    confidence = confidences[s_no, start_no, stop_no].item()

                    top_spans.append((prediction, confidence, start_no, stop_no))

            # print(top_spans)
            top_spans.sort(key=lambda x:x[1], reverse=True)
            # top_spans = [sorted(top_span, reverse=True, key=lambda x: x[1]) for top_span in top_spans]
            # print(top_spans)

            already_tagged = []
            for (prediction, confidence, start_no, stop_no) in top_spans:

                if prediction != 0:

                    conflict = False
                    for i in range(start_no, stop_no+1):
                        if i in already_tagged: conflict = True

                    if not conflict:

                        tag = self.tag_dictionary.get_item_for_index(prediction)

                        span = ''
                        if start_no == stop_no:

                            sentence[start_no].set_label(label_name, 'S-' + tag)
                            span += ' ' + sentence[start_no].text
                            already_tagged.append(start_no)

                        else:
                            for i in range(start_no, stop_no+1):
                                label = 'B-' + tag if i == start_no else 'I-' + tag
                                sentence[i].set_label(label_name, label)
                                span += ' ' + sentence[i].text
                                already_tagged.append(i)

                        # print(f'Added {span} as {tag} with confidence {confidence}')

    def _requires_span_F1_evaluation(self) -> bool:
        return True

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size=32,
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
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if
        you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.
        'gpu' to store embeddings in GPU memory.
        """
        if label_name == None:
            label_name = self.tag_type

        # print(f'predicting: {label_name}')

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

                feature, loss = self.forward_loss(batch, also_return_logits=True)

                if return_loss:
                    overall_loss += loss

                self._obtain_labels(feature=feature, batch_sentences=batch, label_name=label_name)

                # for (sentence, sent_tags) in zip(batch, tags):
                #     for (token, tag) in zip(sentence.tokens, sent_tags):
                #         token.add_tag_label(label_name, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                # for (sentence, sent_all_tags) in zip(batch, all_tags):
                #     for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                #         token.add_tags_proba_dist(label_name, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss / batch_no

    def evaluate(
        self,
        sentences: Union[List[Sentence], Dataset],
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: int = 8,
    ) -> (Result, float):
        eval_loss = 0

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        batch_no: int = 0
        self.beta = 1
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

                # make list of predicted tags
                predicted_spans = sentence.get_spans("predicted")
                predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]

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

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "embeddings": self.embeddings,
            "hidden_size": self.hidden_size,
            "biaffine_size": self.biaffine_size,
            "tag_dictionary": self.tag_dictionary,
            "tag_type": self.tag_type,
            "rnn_layers": self.rnn_layers,
            "use_dropout": self.use_dropout,
            "use_word_dropout": self.use_word_dropout,
            "use_locked_dropout": self.use_locked_dropout,
            "rnn_type": self.rnn_type,
            "reproject_embeddings": self.reproject_embeddings,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        model = BiaffineTagger(
            hidden_size=state["hidden_size"],
            biaffine_size=state["biaffine_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            rnn_layers=state["rnn_layers"],
            dropout=state["use_dropout"],
            word_dropout=state["use_word_dropout"],
            locked_dropout=state["use_locked_dropout"],
            rnn_type=state["rnn_type"],
            reproject_embeddings=state["reproject_embeddings"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def _make_padded_embedding_tensor(self, sentences):
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
        return lengths, sentence_tensor

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                f"Ignore {len(sentences) - len(filtered_sentences)} sentence(s) with no tokens."
            )
        return filtered_sentences

    def _requires_span_F1_evaluation(self) -> bool:
        span_F1 = False
        for item in self.tag_dictionary.get_items():
            if item.startswith('B-'):
                span_F1 = True
        return span_F1

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

                # make list of predicted tags
                predicted_spans = sentence.get_spans("predicted")
                predicted_tags = [(span.tag, repr(span)) for span in predicted_spans]

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