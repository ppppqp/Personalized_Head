from random import randint

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from torch.nn import GRUCell, GRU, LSTMCell

import flair
import random


class MemoryLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            memory_size,
            word_drop_probability: float = 0.2,
            bias: bool = True,
            batch_first: bool = False,
            bidirectional: bool = False,
    ):
        super().__init__()

        # internal variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        # self.word_drop_probability = word_drop_probability
        # self.word_drop = flair.nn.WordDropout(word_drop_probability)

        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.output_size = self.hidden_size

        # initialize cell of forward LSTM
        self.forward_cell = LSTMCell(
            input_size, hidden_size, bias,
        )

        # map to record the last state of each word
        self.word_memory_forward = {}
        self.backward_store = {}
        self.forward_store = {}
        self.word_memory_count = {}

        # if bidirectional, initialize backward cell as well
        if bidirectional:

            self.output_size *= 2

            independent_rnns = True
            # if the backward pass is independent
            backwards_input_size = input_size if independent_rnns else hidden_size

            self.backward_cell = LSTMCell(
                backwards_input_size, hidden_size, bias,
            )

            # map to record the last state of each word in backward pass
            self.word_memory_backward = {}

        # default empty states
        self.initial_word = torch.zeros(input_size, device=flair.device).unsqueeze(0)

        # default hidden and cell states
        self.initial_hidden = torch.zeros(hidden_size, device=flair.device).unsqueeze(0)
        self.initial_cell = torch.zeros(hidden_size, device=flair.device).unsqueeze(0)

        # default memory cell state
        self.initial_memory = torch.zeros(memory_size, device=flair.device)

    def train(self, mode=True):
        super().train(mode=mode)

        print(
            f"Memsize: {len(self.word_memory_forward)}"
        )
        # print(
        #     f"Memsize: {len(self.word_memory_count)}"
        # )

        if mode:
            self.word_memory_forward = self.forward_store.copy()
            if self.bidirectional:
                self.word_memory_backward = self.backward_store.copy()

            # delete_keys = []
            # for key in self.word_memory_forward.keys():
            #     if randint(1, 10) == 1:
            #         delete_keys.append(key)
            #
            # for key in delete_keys:
            #     del self.word_memory_forward[key]
                    # print(key)
            print(
                f"Train mode - resetting memory. Memsize: {len(self.word_memory_forward)}"
            )
            self.word_memory_count = {}
        else:
            self.forward_store = self.word_memory_forward.copy()
            if self.bidirectional:
                self.backward_store = self.word_memory_backward.copy()
            # memory is wiped each time we do evaluation
            print(f"Prediction mode - growing memory. Memsize: {len(self.word_memory_forward)}")

        # if mode:
        #     self.word_memory_forward = {}
        #     if self.bidirectional:
        #         self.word_memory_backward = {}
        #     print(
        #         f"Train mode - resetting memory."
        #     )
        # else:
        #     # memory is wiped each time we do evaluation
        #     print(f"Prediction mode - growing memory")

    def memory_rnn_pass(self, rnn_cell, memory, input_, batch_dp_names, hidden=None, is_forward: bool = True):
        outputs = []

        if hidden is not None:
            if input_.size(0) < hidden.size(0):
                hidden = hidden[:input_.size(0)]

        word_dim = 1 if self.batch_first else 0
        sentence_dim = 0 if self.batch_first else 1

        last_hidden = []

        # go through each sentence
        sentence_indices = list(range(input_.size(sentence_dim)))

        if not is_forward:
            sentence_indices = reversed(sentence_indices)

        for sentence in sentence_indices:
            # print(f'sentence: {sentence}')

            word_indices = list(range(input_.size(word_dim)))
            if not is_forward:
                word_indices = reversed(word_indices)

            # get the previous hidden and cell state (initialize to zero if none)
            previous_hidden_state = self.initial_hidden if hidden is None else hidden[sentence].unsqueeze(0)[
                0]  # input, (h_0, c_0)
            previous_cell_state = self.initial_cell if hidden is None else hidden[sentence].unsqueeze(0)[1]

            # print(row)
            column_outputs = []
            # print(input_.size(word_dim))
            for word in word_indices:
                # print(word)

                if word >= len(batch_dp_names[sentence]):
                    column_outputs.append(self.initial_hidden)

                else:
                    token_text = batch_dp_names[sentence][word]
                    # print(token_text)

                    # get the embedding of the data point (input to RNN)
                    if self.batch_first:
                        data_point = input_[sentence, word].unsqueeze(0)
                    else:
                        data_point = input_[word, sentence].unsqueeze(0)

                    if token_text not in memory:
                        memory[token_text] = self.initial_memory

                    # get the last cell state of this data point
                    word_memory_cell_state = memory[token_text]

                    # TODO: check if this makes sense
                    # word_memory_cell_state = F.normalize(word_memory_cell_state, p=2, dim=0)

                    previous_cell_size = self.hidden_size - self.memory_size

                    # cell state is combination of previous cell state, and last cell state of this word
                    previous_cell_state = torch.cat(
                        [
                            previous_cell_state[0, 0: previous_cell_size],
                            word_memory_cell_state
                        ]
                    ).unsqueeze(0)

                    # if token_text == 'if':
                    #     print()
                    #     m = nn.Tanh()
                    #     print(previous_cell_state[0, 0: previous_cell_size])
                    #     print(m.forward(previous_cell_state[0, 0: previous_cell_size]))
                    #     print(word_memory_cell_state)
                    #     print(m.forward(word_memory_cell_state))
                    #     print(previous_cell_state)
                    #     print(m.forward(previous_cell_state))
                    #     print()

                    input_states = (previous_hidden_state, previous_cell_state)  # input, (h_0, c_0)

                    # send through RNN
                    hidden_state_out = rnn_cell(data_point, input_states)

                    previous_cell_state = hidden_state_out[1]
                    previous_hidden_state = hidden_state_out[0]

                    # split off memory portion and store
                    memory_hidden = previous_cell_state[0, previous_cell_size:previous_cell_size + self.memory_size]
                    memory[token_text] = memory_hidden

                    column_outputs.append(previous_hidden_state)

            last_hidden.append(previous_hidden_state)

            # print(column_outputs)

            if not is_forward:
                column_outputs = list(reversed(column_outputs))

            stack = torch.stack(column_outputs, dim=word_dim)
            # print(stack.size())
            outputs.append(stack)

        if not is_forward:
            outputs = list(reversed(outputs))

        hidden_states_for_all_positions = torch.cat(outputs, dim=sentence_dim)

        all_last_hidden = torch.cat(last_hidden, dim=0)

        return hidden_states_for_all_positions, all_last_hidden

    def forward(self, input_, batch_dp_names, hidden=None):

        # input_ = self.word_drop(input_)

        if self.bidirectional:

            for key in self.word_memory_backward.keys():
                self.word_memory_backward[key] = self.word_memory_backward[key].detach()

            backward, all_last_hidden = self.memory_rnn_pass(self.backward_cell, self.word_memory_backward, input_,
                                                             batch_dp_names, hidden, is_forward=False)

            # detach all hidden states in memory
            # for key in self.word_memory_forward.keys():
            #     self.word_memory_forward[key] = self.word_memory_forward[key].detach()

            forward, all_last_hidden = self.memory_rnn_pass(self.forward_cell, self.word_memory_forward, input_,
                                                            batch_dp_names, hidden, is_forward=True)

            hidden_states_for_all_positions = torch.cat([forward, backward], dim=2)
        else:
            hidden_states_for_all_positions, all_last_hidden = self.memory_rnn_pass(self.forward_cell,
                                                                                    self.word_memory_forward, input_,
                                                                                    batch_dp_names, hidden,
                                                                                    is_forward=True)

        delete_keys = {}
        if self.training:
            for batch in batch_dp_names:
                for text in batch:
                    # word_count = 1 if text not in self.word_memory_count else self.word_memory_count[text] + 1
                    # self.word_memory_count[text] = word_count
                    delete_keys[text] = 1.

        # detach all hidden states in memory
        for key in delete_keys.keys():
            self.word_memory_forward[key] = self.word_memory_forward[key].detach()

        return hidden_states_for_all_positions, all_last_hidden