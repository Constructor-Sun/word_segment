from turtle import forward
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import *
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from dataloader import Sentence

"""
class CWS(BertPreTrainedModel):

    def __init__(self, config):
        super(CWS, self).__init__(config)
        self.bert = BertModel(config)
        self.embedding_dim = 768
        self.hidden_dim = 256
        # self.vocab_size = vocab_size
        # self.tag2id = tag2id
        # self.tagset_size = len(tag2id)
        self.tagset_size = 4
        # self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        self.crf = CRF(4, batch_first=True)

        self.init_weights()

    

    # def init_hidden(self, batch_size, device):
    #     return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
    #             torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

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
"""

class CWS(BertPreTrainedModel):
    def __init__(self, config):
        super(CWS, self).__init__(config)
        self.num_labels = 4

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)

        self.bilstm = nn.LSTM(
            input_size=768,  # 768
            hidden_size=1024 // 2,  # 1024 / 2
            batch_first=True,
            num_layers=2,
            dropout=0.5,  # 0.5
            bidirectional=True
        )

        self.hidden2tag = nn.Linear(1024, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)
        # 得到判别值
        logits = self.hidden2tag(lstm_output)
        
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            # print("logits: ", logits.shape)
            # print("labels: ", labels.shape)
            # print("loss_mask: ", loss_mask.shape)
            loss = -self.crf(logits, labels, loss_mask)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
