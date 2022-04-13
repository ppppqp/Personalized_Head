import logging
from pathlib import Path
from typing import List, Union, Callable, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import flair.nn
import flair.embeddings
from flair.data import Dictionary, Sentence, Label, Token, space_tokenizer, DataPair
from flair.datasets import SentenceDataset, StringDataset
from flair.file_utils import cached_path
from flair.training_utils import (
    convert_labels_to_one_hot,
    Metric,
    Result,
    store_embeddings,
)

log = logging.getLogger("flair")


class PairClassifier(flair.nn.Model):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an RNN to obtain a text representation, and puts the
    text representation in the end into a linear layer to get the actual class label.
    The model can handle single and multi class data sets.
    """

    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_dictionary: Dictionary,
    ):
        """
        Initializes a TextClassifier
        :param document_embeddings: embeddings used to embed each data point
        :param label_dictionary: dictionary of labels you want to predict
        :param multi_label: auto-detected by default, but you can set this to True to force multi-label prediction
        or False to force single-label prediction
        :param multi_label_threshold: If multi-label you can set the threshold to make predictions
        :param beta: Parameter for F-beta score for evaluation and training annealing
        :param loss_weights: Dictionary of weights for labels for the loss function
        (if any label's weight is unspecified it will default to 1.0)
        """

        super(PairClassifier, self).__init__()

        self.document_embeddings: flair.embeddings.DocumentRNNEmbeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary

        self.decoder = nn.Linear(
            self.document_embeddings.embedding_length * 2, len(self.label_dictionary)
        )

        self._init_weights()

        self.loss_function = nn.CrossEntropyLoss()

        # auto-spawn on GPU if available
        self.to(flair.device)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, sentences: List[DataPair]) -> List[List[float]]:

        first_sentences = [pair.first for pair in sentences]
        second_sentences = [pair.second for pair in sentences]

        self.document_embeddings.embed(first_sentences)
        self.document_embeddings.embed(second_sentences)

        text_embedding_list = [
            pair.embedding.unsqueeze(0) for pair in sentences
        ]
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)

        label_scores = self.decoder(text_embedding_tensor)

        return label_scores

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        model = PairClassifier(
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
        )

        model.load_state_dict(state["state_dict"])
        return model

    def forward_loss(
        self, data_points: Union[List[Sentence], Sentence]
    ) -> torch.tensor:

        scores = self.forward(data_points)
        return self._calculate_loss(scores, data_points)

    def forward_labels_and_loss(
        self, sentences: Union[Sentence, List[Sentence]]
    ) -> (List[List[Label]], torch.tensor):
        scores = self.forward(sentences)
        labels = self._obtain_labels(scores)
        loss = self._calculate_loss(scores, sentences)
        return labels, loss

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size: int = 32,
        embedding_storage_mode="none",
        multi_class_prob: bool = False,
        verbose: bool = False,
        use_tokenizer: Union[bool, Callable[[str], List[Token]]] = space_tokenizer,
    ) -> List[Sentence]:
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param embedding_storage_mode: 'none' for the minimum memory footprint, 'cpu' to store embeddings in Ram,
        'gpu' to store embeddings in GPU memory.
        :param multi_class_prob : return probability for all class for multiclass
        :param verbose: set to True to display a progress bar
        :param use_tokenizer: a custom tokenizer when string are provided (default is space based tokenizer).
        :return: the list of sentences containing the labels
        """
        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence) or isinstance(sentences, str):
                sentences = [sentences]

            if (flair.device.type == "cuda") and embedding_storage_mode == "cpu":
                log.warning(
                    "You are inferring on GPU with parameter 'embedding_storage_mode' set to 'cpu'."
                    "This option will slow down your inference, usually 'none' (default value) "
                    "is a better choice."
                )

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )
            original_order_index = sorted(
                range(len(rev_order_len_index)), key=lambda k: rev_order_len_index[k]
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            if isinstance(sentences[0], DataPair):
                # remove previous embeddings
                store_embeddings(reordered_sentences, "none")
                dataset = SentenceDataset(reordered_sentences)
            else:
                dataset = StringDataset(
                    reordered_sentences, use_tokenizer=use_tokenizer
                )
            dataloader = DataLoader(
                dataset=dataset, batch_size=mini_batch_size, collate_fn=lambda x: x
            )

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            results: List[Sentence] = []
            for i, batch in enumerate(dataloader):
                if verbose:
                    dataloader.set_description(f"Inferencing on batch {i}")
                results += batch
                # batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                scores = self.forward(batch)
                predicted_labels = self._obtain_labels(
                    scores, predict_prob=multi_class_prob
                )

                for (sentence, labels) in zip(batch, predicted_labels):
                    sentence.labels = labels

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embedding_storage_mode)

            results: List[Union[Sentence, str]] = [
                results[index] for index in original_order_index
            ]
            assert len(sentences) == len(results)
            return results

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):

        with torch.no_grad():
            eval_loss = 0

            metric = Metric("Evaluation")

            lines: List[str] = []
            batch_count: int = 0
            for batch in data_loader:

                batch_count += 1

                labels, loss = self.forward_labels_and_loss(batch)

                eval_loss += loss

                sentence_one_for_batch = [sent.first.to_plain_string() for sent in batch]
                sentence_two_for_batch = [sent.second.to_plain_string() for sent in batch]
                confidences_for_batch = [
                    [label.score for label in sent_labels] for sent_labels in labels
                ]
                predictions_for_batch = [
                    [label.value for label in sent_labels] for sent_labels in labels
                ]
                true_values_for_batch = [
                    sentence.get_label_names() for sentence in batch
                ]
                available_labels = self.label_dictionary.get_items()

                for sentence_one, sentence_two, confidence, prediction, true_value in zip(
                    sentence_one_for_batch,
                    sentence_two_for_batch,
                    confidences_for_batch,
                    predictions_for_batch,
                    true_values_for_batch,
                ):
                    eval_line = "{}\t{}\t{}\t{}\t{}\n".format(
                        sentence_one, sentence_two, true_value, prediction, confidence
                    )
                    lines.append(eval_line)

                for predictions_for_sentence, true_values_for_sentence in zip(
                    predictions_for_batch, true_values_for_batch
                ):

                    for label in available_labels:
                        if (
                            label in predictions_for_sentence
                            and label in true_values_for_sentence
                        ):
                            metric.add_tp(label)
                        elif (
                            label in predictions_for_sentence
                            and label not in true_values_for_sentence
                        ):
                            metric.add_fp(label)
                        elif (
                            label not in predictions_for_sentence
                            and label in true_values_for_sentence
                        ):
                            metric.add_fn(label)
                        elif (
                            label not in predictions_for_sentence
                            and label not in true_values_for_sentence
                        ):
                            metric.add_tn(label)

                store_embeddings(batch, embedding_storage_mode)

            eval_loss /= batch_count

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            return result, eval_loss

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                "Ignore {} sentence(s) with no tokens.".format(
                    len(sentences) - len(filtered_sentences)
                )
            )
        return filtered_sentences

    def _calculate_loss(
        self, scores: torch.tensor, sentences: List[Sentence]
    ) -> torch.tensor:
        """
        Calculates the loss.
        :param scores: the prediction scores from the model
        :param sentences: list of sentences
        :return: loss value
        """
        return self._calculate_single_label_loss(scores, sentences)

    def _obtain_labels(
        self, scores: List[List[float]], predict_prob: bool = False
    ) -> List[List[Label]]:
        """
        Predicts the labels of sentences.
        :param scores: the prediction scores from the model
        :return: list of predicted labels
        """
        return [self._get_single_label(s) for s in scores]

    def _get_single_label(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        conf, idx = torch.max(softmax, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [Label(label, conf.item())]

    def _calculate_single_label_loss(
        self, label_scores, sentences: List[Sentence]
    ) -> float:
        return self.loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, sentences: List[Sentence]):
        label_list = [sentence.get_label_names() for sentence in sentences]
        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(flair.device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.LongTensor(
                [
                    self.label_dictionary.get_idx_for_item(label.value)
                    for label in sentence.labels
                ]
            )
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0).to(flair.device)

        return vec