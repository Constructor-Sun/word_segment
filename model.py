from turtle import forward
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import *
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from dataloader import Sentence


class CWS(BertPreTrainedModel):

    def __init__(self, embedding_dim, hidden_dim, config):
        super(CWS, self).__init__(config)
        self.bert = BertModel.from_pretrained(config)
        # self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        # self.tag2id = tag2id
        # self.tagset_size = len(tag2id)

        # self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(4, batch_first=True)

        self.init_weights()

    """

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    """

    def _get_lstm_features(self, sentence):
        # batch_size, seq_len = sentence.shape[0], sentence.shape[1]

        # idx->embedding
        # embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1)
        # embeds = pack_padded_sequence(embeds, length, batch_first=True)
        padded_sequence_output = pad_sequence(sentence, batch_first = True)

        # LSTM forward
        # self.hidden = self.init_hidden(batch_size, sentence.device)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, self.hidden = self.lstm(padded_sequence_output)
        # lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, input_data, attention_mask, label_masks, tags):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        sequence_output = outputs[0]
        sentence = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(sequence_output, input_token_starts)]
        
        emissions = self._get_lstm_features(sentence)
        loss = -self.crf(emissions, tags, label_masks) # reduction='mean'
        return loss

    def infer(self, input_data, attention_mask, label_masks):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        sequence_output = outputs[0]
        sentence = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(sequence_output, input_token_starts)]
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, label_masks)
