import math
from collections import Counter

import torch
from typing import List, Union

from torch.nn.parameter import Parameter

import flair
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentEmbeddings
from flair.nn import LockedDropout, WordDropout
import time


class Stopwatch(object):
    """A stopwatch utility for timing execution that can be used as a regular
    object or as a context manager.
    NOTE: This should not be used an accurate benchmark of Python code, but a
    way to check how much time has elapsed between actions. And this does not
    account for changes or blips in the system clock.
    Instance attributes:
    start_time -- timestamp when the timer started
    stop_time -- timestamp when the timer stopped
    As a regular object:
    >>> stopwatch = Stopwatch()
    >>> stopwatch.start()
    >>> time.sleep(1)
    >>> 1 <= stopwatch.time_elapsed <= 2
    True
    >>> time.sleep(1)
    >>> stopwatch.stop()
    >>> 2 <= stopwatch.total_run_time
    True
    As a context manager:
    >>> with Stopwatch() as stopwatch:
    ...     time.sleep(1)
    ...     print repr(1 <= stopwatch.time_elapsed <= 2)
    ...     time.sleep(1)
    True
    >>> 2 <= stopwatch.total_run_time
    True
    """

    def __init__(self):
        """Initialize a new `Stopwatch`, but do not start timing."""
        self.start_time = None
        self.stop_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def stop(self):
        """Stop timing."""
        self.stop_time = time.time()

    @property
    def time_elapsed(self):
        """Return the number of seconds that have elapsed since this
        `Stopwatch` started timing.
        This is used for checking how much time has elapsed while the timer is
        still running.
        """
        assert not self.stop_time, \
            "Can't check `time_elapsed` on an ended `Stopwatch`."
        return time.time() - self.start_time

    @property
    def total_run_time(self):
        """Return the number of seconds that elapsed from when this `Stopwatch`
        started to when it ended.
        """
        return self.stop_time - self.start_time

    def __enter__(self):
        """Start timing and return this `Stopwatch` instance."""
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        """Stop timing.
        If there was an exception inside the `with` block, re-raise it.
        >>> with Stopwatch() as stopwatch:
        ...     raise Exception
        Traceback (most recent call last):
            ...
        Exception
        """
        self.stop()


class DocumentRNNEmbeddingsFast(DocumentEmbeddings):
    def __init__(
            self,
            embeddings: List[TokenEmbeddings],
            hidden_size=128,
            rnn_layers=1,
            reproject_words: bool = True,
            reproject_words_dimension: int = None,
            bidirectional: bool = False,
            dropout: float = 0.5,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            rnn_type="GRU",
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        :param rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.rnn_type = rnn_type

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )

        # bidirectional RNN on top of embedding layer
        if rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
            )
        else:
            self.rnn = torch.nn.GRU(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
            )

        self.name = "document_" + self.rnn._get_name()

        # dropouts
        if locked_dropout > 0.0:
            self.dropout: torch.nn.Module = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)

        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        # first, sort sentences by number of tokens
        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        # go through each sentence in batch
        for i, sentence in enumerate(sentences):
            lengths.append(len(sentence.tokens))

            sentence_tensor = torch.cat([token.get_embedding().unsqueeze(0) for token in sentence], 0).to(flair.device)

            # ADD TO SENTENCE LIST: add the representation
            all_sentence_tensors.append(sentence_tensor)

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        # use word dropout if set
        # if self.use_word_dropout:
        #     sentence_tensor = self.word_dropout(sentence_tensor)

        # if self.reproject_words:
        #     sentence_tensor = self.word_reprojection_map(sentence_tensor)

        # sentence_tensor = self.dropout(sentence_tensor)

        packed = torch.nn.utils.rnn.pack_sequence(all_sentence_tensors, lengths)

        self.rnn.flatten_parameters()

        rnn_out, hidden = self.rnn(packed)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_out)

        outputs = self.dropout(outputs)

        # --------------------------------------------------------------------
        # EXTRACT EMBEDDINGS FROM RNN
        # --------------------------------------------------------------------
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[length - 1, sentence_no]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[0, sentence_no]
                embedding = torch.cat([first_rep, last_rep], 0)

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


class MemoryEmbeddingsFast(TokenEmbeddings):
    def __init__(
            self,
            contextual_embeddings: TokenEmbeddings,
            concat_word_embeddings: bool = True,
            hidden_states: int = 64,
            reproject: bool = True,
            max_memory_length: int = 8,
            dropout: float = 0.5,
            **kwargs,
    ):

        super().__init__()

        # variables
        self.hidden_states = hidden_states
        self.max_memory_length: int = max_memory_length
        self.effective_memory_length: int = self.max_memory_length
        self.concat_word_embeddings: bool = concat_word_embeddings
        self.static_embeddings: bool = False

        # use the character language model embeddings as basis
        if type(contextual_embeddings) is str:
            self.context_embeddings: FlairEmbeddings = FlairEmbeddings(
                contextual_embeddings, **kwargs
            )
        else:
            self.context_embeddings: TokenEmbeddings = contextual_embeddings

        self.sub_embedding_names = (
            [emb.name for emb in self.context_embeddings.embeddings]
            if type(self.context_embeddings) is StackedEmbeddings
            else [self.context_embeddings.name]
        )

        # determine embedding length
        self.__embedding_length = (
            self.context_embeddings.embedding_length + self.hidden_states
            if self.concat_word_embeddings
            else 0 + self.hidden_states
        )

        # the memory
        self.word_history = {}
        self.state_history = {}

        # the NN
        self.dropout = torch.nn.Dropout(dropout)

        self.reproject: bool = reproject
        if self.reproject:
            self.reprojection_layer = torch.nn.Linear(
                self.context_embeddings.embedding_length,
                self.context_embeddings.embedding_length,
            )

        self.rnn = torch.nn.GRU(
            self.context_embeddings.embedding_length,
            self.hidden_states,
            num_layers=1,
            bidirectional=False,
        )
        self.name = self.context_embeddings.name + "-memory"
        #
        self.all_time = 0.
        self.prepare_time = 0.
        self.nn_time = 0.
        self.add_to_memory_time = 0.
        self.add_to_words_time = 0.
        self.pad_time = 0.
        self.rest_time = 0.
        self.batch_count = 0

        self.to(flair.device)

    def train(self, mode=True):
        super().train(mode=mode)
        compress = False
        if mode:
            # memory is wiped each time we do a training run
            print("train mode resetting embeddings")
            self.word_history = {}
            self.state_history = {}
            print(self.word_history)
            print(self.state_history)
            self.effective_memory_length = self.max_memory_length
            print(self.effective_memory_length)
        elif compress:
            # memory is wiped each time we do a training run
            print("prediction mode no backprop")
            self.word_history = {}
            for word in self.state_history:
                self.state_history[word] = [self.state_history[word][-1].clone()]
            self.effective_memory_length = 1

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        self.context_embeddings.embed(sentences)
        #
        # all_stopwatch = Stopwatch()
        # prepare_stopwatch = Stopwatch()
        # nn_stopwatch = Stopwatch()
        # add_to_memory_stopwatch = Stopwatch()
        # add_to_words_stopwatch = Stopwatch()
        # pad_stopwatch = Stopwatch()
        # rest_stopwatch = Stopwatch()
        #
        # all_stopwatch.start()

        # padding = torch.zeros(
        #     self.context_embeddings.embedding_length, dtype=torch.float
        # ).to(flair.device)

        # determine and add to history of each token
        surface_form_history = {}
        for sentence in sentences:
            for token in sentence:
                # add current embedding to the memory
                local_embedding = token.get_embedding(
                    self.sub_embedding_names
                )

                if token.text not in self.word_history:
                    self.word_history[token.text] = [local_embedding]
                else:
                    self.word_history[token.text].append(local_embedding)

                surface_form_history[token.text] = self.word_history[token.text]

        # sort surface forms by longest history
        surface_forms_sorted_by_memory_length = sorted(surface_form_history,
                                                       key=lambda k: len(surface_form_history[k]),
                                                       reverse=True)
        logest_memory_length_in_batch = len(surface_form_history[surface_forms_sorted_by_memory_length[0]])

        # all_surface_form_histories = []

        # initialize zero-padded word embeddings tensor
        all_surface_form_tensor = torch.zeros(
            [
                logest_memory_length_in_batch,
                len(surface_form_history),
                self.context_embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        # default zero-state tensor
        zeros = torch.zeros(self.hidden_states, device=flair.device)

        all_surface_form_initial_hidden = []
        lengths = []

        # prepare_stopwatch.start()

        # print('all:')
        # print(all_surface_form_tensor)
        # go through each unique surface form
        for s_id, surface_form in enumerate(surface_forms_sorted_by_memory_length):

            # pad_stopwatch.start()

            # get embedding history of this surface form and bring to flair.device
            surface_form_embedding_history = surface_form_history[surface_form]

            # for i in range(len(surface_form_embedding_history)):
            #     surface_form_embedding_history[i] = surface_form_embedding_history[i].to(flair.device)

            # pad shorter sequences out
            # for add in range(logest_memory_length_in_batch - len(surface_form_embedding_history)):
            #     surface_form_embedding_history.append(padding)

            for i in range(len(surface_form_embedding_history)):
                # print('histry:')
                # print(surface_form_embedding_history[i])
                all_surface_form_tensor[i, s_id] = surface_form_embedding_history[i]

            # print('all:')
            # print(all_surface_form_tensor)
            # all_surface_form_tensor[s_id][: len(sentence)] = torch.stack(
            #     [token.get_embedding() for token in sentence], 0
            # )

            lengths.append(len(surface_form_embedding_history))

            # pad_stopwatch.stop()
            # self.pad_time += pad_stopwatch.total_run_time
            # rest_stopwatch.start()

            # truncate surface form history if necessary
            window = 0 if len(self.word_history[surface_form]) < self.effective_memory_length else len(
                self.word_history[surface_form]) - self.effective_memory_length
            if window > 0:
                self.word_history[surface_form] = self.word_history[surface_form][
                                                  window:self.effective_memory_length + window]

            # print(surface_form_embedding_history)

            # make padded tensor of history
            # surface_form_embedding_history = torch.stack(surface_form_embedding_history, dim=0)

            # surface_form_embedding_history =  torch.cat(surface_form_embedding_history).view(-1, self.context_embeddings.embedding_length)

            # surface_form_embedding_history = surface_form_embedding_history.to(flair.device)

            # add history to all
            # all_surface_form_histories.append(surface_form_embedding_history
            # )

            # initialize first hidden state if necessary
            if surface_form not in self.state_history:
                self.state_history[surface_form] = [zeros]

            all_surface_form_initial_hidden.append(
                self.state_history[surface_form][0]
            )

            # rest_stopwatch.stop()
            # self.rest_time += rest_stopwatch.total_run_time

        # make batch tensors
        # all_surface_form_histories = torch.stack(all_surface_form_histories, 1)
        all_surface_form_histories = all_surface_form_tensor
        # print(all_surface_form_tensor)
        # asd

        all_surface_form_initial_hidden = torch.stack(
            all_surface_form_initial_hidden, 0
        ).unsqueeze(0)

        # prepare_stopwatch.stop()
        #
        # nn_stopwatch.start()

        # dropout!
        all_surface_form_histories = self.dropout(all_surface_form_histories)
        # print(all_surface_form_histories)

        # reproject if set
        if self.reproject:
            all_surface_form_histories = self.reprojection_layer(
                all_surface_form_histories
            )

        # send through RNN
        # rnn_out, hidden = self.rnn(all_surface_form_histories, all_surface_form_initial_hidden)
        packed = torch.nn.utils.rnn.pack_padded_sequence(all_surface_form_histories, lengths)

        packed_output, hidden = self.rnn(packed, all_surface_form_initial_hidden)

        rnn_out, hidden = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output
        )

        # nn_stopwatch.stop()
        #
        # add_to_memory_stopwatch.start()

        # go through each unique surface form and update state history
        for idx, surface_form in enumerate(surface_forms_sorted_by_memory_length):
            # retrieve hidden states for this surface form, and the last initial hidden state
            hidden_states_of_surface_form = rnn_out[0:lengths[idx], idx]
            last_initial_hidden_state = all_surface_form_initial_hidden[0][idx].unsqueeze(0)

            # concat both to get new hidden state history
            hidden_state_history = torch.cat([last_initial_hidden_state, hidden_states_of_surface_form], dim=0)

            # if history is too long, truncate to window
            window = 0 if hidden_state_history.size(0) < self.effective_memory_length else hidden_state_history.size(
                0) - self.effective_memory_length
            state_history_window = hidden_state_history.detach()[window:self.effective_memory_length + window]

            # set as new state history of surface form (before: .cpu())
            self.state_history[surface_form] = state_history_window

        # add_to_memory_stopwatch.stop()
        #
        # add_to_words_stopwatch.start()

        # finally, go through each token of each sentence and set the embedding
        for sentence in sentences:

            for token in sentence:

                idx = surface_forms_sorted_by_memory_length.index(token.text)

                embedding = rnn_out[lengths[idx] - 1, idx]

                # print(self.state_history[token.text][-1])

                token.set_embedding(self.name, embedding)

                if not self.concat_word_embeddings:
                    for subembedding in self.sub_embedding_names:
                        if subembedding in token._embeddings.keys():
                            del token._embeddings[subembedding]

        # add_to_words_stopwatch.stop()
        # all_stopwatch.stop()
        # self.all_time += all_stopwatch.total_run_time
        # self.prepare_time += prepare_stopwatch.total_run_time
        # self.nn_time += nn_stopwatch.total_run_time
        # self.add_to_memory_time += add_to_memory_stopwatch.total_run_time
        # self.add_to_words_time += add_to_words_stopwatch.total_run_time
        # self.batch_count += 1
        # print(f'Total time    : {self.all_time}')
        # print(f' - Preparation: {round(self.prepare_time / self.all_time, 2)}% ({self.prepare_time})')
        # print(f'      - Pad   : {round(self.pad_time / self.prepare_time, 2)}% ({self.pad_time})')
        # print(f'      - Rest  : {round(self.rest_time / self.prepare_time, 2)}% ({self.rest_time})')
        # print(f' - NN + RNN   : {round(self.nn_time / self.all_time, 2)}% ({self.nn_time})')
        # print(f' - Memory     : {round(self.add_to_memory_time / self.all_time, 2)}% ({self.add_to_memory_time})')
        # print(f' - Add Words  : {round(self.add_to_words_time / self.all_time, 2)}% ({self.add_to_words_time})')
        # print(f'Batches: {self.batch_count}')
        # print()
        #
        # if self.all_time > 10:
        #     asd

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        names = [self.name]
        names.extend(self.context_embeddings.get_names())
        return names


class MemoryEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            contextual_embeddings: TokenEmbeddings,
            concat_word_embeddings: bool = True,
            hidden_states: int = 64,
            reproject: bool = True,
            max_memory_length: int = 8,
            dropout: float = 0.5,
            use_batch_memory_state: bool = False,
    ):

        super().__init__()

        # variables
        self.hidden_states = hidden_states
        self.max_memory_length: int = max_memory_length
        self.effective_memory_length: int = self.max_memory_length
        self.concat_word_embeddings: bool = concat_word_embeddings
        self.static_embeddings: bool = False

        self.context_embeddings: TokenEmbeddings = contextual_embeddings

        self.sub_embedding_names = (
            [emb.name for emb in self.context_embeddings.embeddings]
            if type(self.context_embeddings) is StackedEmbeddings
            else [self.context_embeddings.name]
        )

        # determine embedding length
        self.__embedding_length = (
            self.context_embeddings.embedding_length + self.hidden_states
            if self.concat_word_embeddings
            else 0 + self.hidden_states
        )

        # the memory
        self.word_history = {}
        self.state_history = {}

        # the NN
        self.dropout = torch.nn.Dropout(dropout)

        self.reproject: bool = reproject
        if self.reproject:
            self.reprojection_layer = torch.nn.Linear(
                self.context_embeddings.embedding_length,
                self.context_embeddings.embedding_length,
            )

        self.rnn = torch.nn.GRU(
            self.context_embeddings.embedding_length,
            self.hidden_states,
            num_layers=1,
            bidirectional=False,
        )
        self.name = self.context_embeddings.name + "-memory"

        self.use_batch_memory_state = use_batch_memory_state

        self.to(flair.device)

    def train(self, mode=True):
        super().train(mode=mode)
        compress = False
        if mode:
            # memory is wiped each time we do a training run
            print("train mode resetting embeddings")
            self.word_history = {}
            self.state_history = {}
            print(self.word_history)
            print(self.state_history)
            self.effective_memory_length = self.max_memory_length
            print(self.effective_memory_length)
        elif compress:
            # memory is wiped each time we do a training run
            print("prediction mode no backprop")
            self.word_history = {}
            for word in self.state_history:
                self.state_history[word] = [self.state_history[word][-1].clone()]
            self.effective_memory_length = 1

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        self.context_embeddings.embed(sentences)

        # determine and add to history of each token
        surface_form_history = {}
        for sentence in sentences:
            for token in sentence:
                # add current embedding to the memory
                local_embedding = token.get_embedding(
                    self.sub_embedding_names
                )

                if token.text not in self.word_history:
                    self.word_history[token.text] = [local_embedding]
                else:
                    self.word_history[token.text].append(local_embedding)

                surface_form_history[token.text] = self.word_history[token.text]

        # sort surface forms by longest history
        surface_forms_sorted_by_memory_length = sorted(surface_form_history,
                                                       key=lambda k: len(surface_form_history[k]),
                                                       reverse=True)
        logest_memory_length_in_batch = len(surface_form_history[surface_forms_sorted_by_memory_length[0]])
        # print(surface_forms_sorted_by_memory_length)

        # initialize zero-padded word embeddings tensor
        all_surface_form_tensor = torch.zeros(
            [
                logest_memory_length_in_batch,
                len(surface_form_history),
                self.context_embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )
        # default zero-state tensor
        zeros = torch.zeros(self.hidden_states, device=flair.device)

        all_surface_form_initial_hidden = []
        lengths = []

        # go through each unique surface form
        for s_id, surface_form in enumerate(surface_forms_sorted_by_memory_length):

            # get embedding history of this surface form and bring to flair.device
            surface_form_embedding_history = surface_form_history[surface_form]
            for i in range(len(surface_form_embedding_history)):
                # TODO: use dropout only on first one
                all_surface_form_tensor[i, s_id] = surface_form_embedding_history[i]

            lengths.append(len(surface_form_embedding_history))

            # truncate surface form history if necessary
            window = 0 if len(self.word_history[surface_form]) < self.effective_memory_length else len(
                self.word_history[surface_form]) - self.effective_memory_length
            if window > 0:
                self.word_history[surface_form] = self.word_history[surface_form][
                                                  window:self.effective_memory_length + window]

            # initialize first hidden state if necessary
            if surface_form not in self.state_history:
                self.state_history[surface_form] = zeros

            all_surface_form_initial_hidden.append(
                self.state_history[surface_form]
            )

        # make batch tensors
        all_surface_form_histories = all_surface_form_tensor

        all_surface_form_initial_hidden = torch.stack(
            all_surface_form_initial_hidden, 0
        ).unsqueeze(0)

        # print(all_surface_form_initial_hidden)
        # print(all_surface_form_initial_hidden.size())

        # dropout!
        all_surface_form_histories = self.dropout(all_surface_form_histories)

        # reproject if set
        if self.reproject:
            all_surface_form_histories = self.reprojection_layer(
                all_surface_form_histories
            )

        # send through RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(all_surface_form_histories, lengths)

        packed_output, hidden = self.rnn(packed, all_surface_form_initial_hidden)

        rnn_out, hidden = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output
        )

        # print('rnn out: ')
        # print(rnn_out.size())
        # print(rnn_out)

        if not self.training:
            # go through each unique surface form and update state history
            for idx, surface_form in enumerate(surface_forms_sorted_by_memory_length):
                # retrieve hidden states for this surface form, and the last initial hidden state
                self.state_history[surface_form] = rnn_out[lengths[idx] - 1, idx]

        # print(self.state_history)

        # finally, go through each token of each sentence and set the embedding
        surface_form_counter = Counter()
        for sentence in sentences:

            for token in sentence:
                surface_form_counter[token.text] += 1

                idx = surface_forms_sorted_by_memory_length.index(token.text)
                if self.use_batch_memory_state:
                    memory_index = lengths[idx] - 1
                else:
                    memory_index = surface_form_counter[token.text] - 1

                embedding = rnn_out[memory_index, idx]

                if self.concat_word_embeddings:
                    import random
                    stack = False if self.training and random.randint(1, 100) <= 50 else True

                    embedding = torch.cat([embedding, token.get_embedding(self.sub_embedding_names)]) if stack \
                        else torch.cat(
                        [embedding, torch.zeros(self.embedding_length - self.hidden_states, device=flair.device)])

                token.set_embedding(self.name, embedding)

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def extra_repr(self):
        return f"[use_batch_memory_state='{self.use_batch_memory_state}', concat_word_embeddings='{self.concat_word_embeddings}']"

    # def get_names(self) -> List[str]:
    #     """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
    #     this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
    #     Then, the list contains the names of all embeddings in the stack."""
    #     names = [self.name]
    #     names.extend(self.context_embeddings.get_names())
    #     return names


class OrderedMemoryEmbeddings(TokenEmbeddings):
    def __init__(
            self,
            contextual_embeddings: TokenEmbeddings,
            concat_word_embeddings: bool = True,
            hidden_states: int = 64,
            reproject: bool = False,
            max_memory_length: int = 8,
            dropout: float = 0.0,
            use_batch_memory_state: bool = False,
            train_initial_hidden_state: bool = True,
    ):

        super().__init__()

        # variables
        self.hidden_states = hidden_states
        self.max_memory_length: int = max_memory_length
        self.effective_memory_length: int = self.max_memory_length
        self.concat_word_embeddings: bool = concat_word_embeddings
        self.static_embeddings: bool = False
        self.train_initial_hidden_state = train_initial_hidden_state

        self.context_embeddings: TokenEmbeddings = contextual_embeddings

        self.sub_embedding_names = (
            [emb.name for emb in self.context_embeddings.embeddings]
            if type(self.context_embeddings) is StackedEmbeddings
            else [self.context_embeddings.name]
        )

        # determine embedding length
        self.__embedding_length = (
            self.context_embeddings.embedding_length + self.hidden_states
            if self.concat_word_embeddings
            else 0 + self.hidden_states
        )

        # the memory
        self.last_instance = {}
        self.last_state = {}

        # the NN
        self.dropout = torch.nn.Dropout(dropout)

        self.reproject: bool = reproject
        if self.reproject:
            self.reprojection_layer = torch.nn.Linear(
                self.context_embeddings.embedding_length,
                self.context_embeddings.embedding_length,
            )

        self.rnn = torch.nn.GRU(
            self.context_embeddings.embedding_length,
            self.hidden_states,
            num_layers=1,
            bidirectional=False,
        )
        if self.train_initial_hidden_state:
            self.initial_hidden = Parameter(torch.zeros(self.hidden_states), requires_grad=True)
        else:
            self.initial_hidden = torch.zeros(self.hidden_states)

        self.name = self.context_embeddings.name + "+memory"

        self.use_batch_memory_state = use_batch_memory_state

        self.to(flair.device)

    def train(self, mode=True):
        super().train(mode=mode)

        if mode:
            # memory is wiped each time we do a training run
            print("train mode resetting embeddings")
            self.last_state = {}
            self.effective_memory_length = self.max_memory_length
            print(self.effective_memory_length)

        else:
            # memory is wiped each time we do a training run
            print("prediction mode no backprop")
            for word in self.last_instance.keys():
                self.last_state[word] = getattr(self.last_instance[word], 'last_hidden')
            self.effective_memory_length = 0

            # print(self.last_state)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        word_history = {}

        self.context_embeddings.embed(sentences)

        # default zero-state tensor
        # zeros = torch.zeros(self.hidden_states, device=flair.device)

        # go through all tokens and set previous instance history if not yet set

        if self.training:
            idx = 0
            for sentence in sentences:
                for token in sentence:

                    if not hasattr(token, 'previous_instance'):
                        if token.text in self.last_instance.keys():
                            setattr(token, 'previous_instance', self.last_instance[token.text])
                        else:
                            setattr(token, 'previous_instance', None)
                    idx += 1

                    self.last_instance[token.text] = token
        else:
            idx = 0
            for sentence in sentences:
                for token in sentence:
                    idx += 1

        # initialize zero-padded word embeddings tensor
        all_surface_form_tensor = torch.zeros(
            [
                self.effective_memory_length + 1,
                idx,
                self.context_embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        all_hidden_tensor = torch.zeros(
            [
                1,
                idx,
                self.hidden_states,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        # print(all_surface_form_tensor.size())

        idx = 0
        lengths = []
        for sentence in sentences:
            for token in sentence:

                word_history[idx] = [token]
                memory = token
                if self.training:
                    while True:
                        memory = memory.previous_instance
                        if memory == None: break
                        if len(word_history[idx]) >= self.effective_memory_length + 1: break

                        word_history[idx].insert(0, memory)

                for m_id, memory_token in enumerate(word_history[idx]):
                    all_surface_form_tensor[m_id, idx] = memory_token.get_embedding(self.sub_embedding_names)

                all_hidden_tensor[0, idx] = \
                    self.initial_hidden if token.text not in self.last_state.keys() \
                    else self.last_state[token.text]

                lengths.append(len(word_history[idx]) -1) # length of history only (do not count current token)

                idx += 1

        # make batch tensors
        all_surface_form_histories = all_surface_form_tensor

        # dropout!
        all_surface_form_histories = self.dropout(all_surface_form_histories)

        # reproject if set
        if self.reproject:
            all_surface_form_histories = self.reprojection_layer(
                all_surface_form_histories
            )

        # print(all_surface_form_histories.size())
        rnn_out, hidden = self.rnn(all_surface_form_histories, all_hidden_tensor)
        # print(rnn_out.size())

        # finally, go through each token of each sentence and set the embedding
        idx = 0
        for sentence in sentences:

            for token in sentence:

                last_memory_index = lengths[idx] - 1
                updated_memory_index = lengths[idx]

                if last_memory_index >= 0:
                    embedding = rnn_out[last_memory_index, idx]
                else:
                    embedding = self.initial_hidden if token.text not in self.last_state.keys() \
                                else self.last_state[token.text]

                # remember last hidden state of token
                last_hidden = rnn_out[updated_memory_index, idx]

                if self.training:
                    setattr(token, 'last_hidden', last_hidden.clone().detach())
                else:
                    self.last_state[token.text] = last_hidden.clone().detach()

                if self.concat_word_embeddings:
                    import random
                    stack = False if self.training and random.randint(1, 100) <= 50 else True

                    embedding = torch.cat([embedding, token.get_embedding(self.sub_embedding_names)]) if stack \
                        else torch.cat(
                        [embedding, torch.zeros(self.embedding_length - self.hidden_states, device=flair.device)])

                token.set_embedding(self.name, embedding)

                idx += 1

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def extra_repr(self):
        return f"[use_batch_memory_state='{self.use_batch_memory_state}', concat_word_embeddings='{self.concat_word_embeddings}']"

    def __getstate__(self):

        for word in self.last_instance.keys():
            self.last_state[word] = getattr(self.last_instance[word], 'last_hidden')
        self.last_instance = {}

        return self.__dict__
