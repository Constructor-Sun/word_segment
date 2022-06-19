import torch
import pickle
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sentence(Dataset):
    def __init__(self, x, y, batch_size=10):
        # self.tokenizer = BertTokenizer.from_pretrained('/content/word_segment/pretrained_bert_models/bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('./word_segment/pretrained_bert_models/bert-base-chinese')
        self.data = self.preprocess(x, y)
        self.batch_size = batch_size
        # self.word_pad_idx = 0

    def __len__(self):
        #return len(self.x)
        return len(self.data)

    def __getitem__(self, idx):
        """
        if (len(self.x[idx]) != len(self.y[idx]) + 2):
            print()
            print("error: ", self.x[idx])
            print("decoded_error:", self.tokenizer.decode(self.x[idx]))
            print("len(decoded_error):", len(self.tokenizer.decode(self.x[idx])))
            print("len(self.x[idx]): ", len(self.x[idx]))
            print("len(self.y[idx]): ", len(self.y[idx]))
            print("self.y[idx]: ", self.y[idx])
            print()
        """
        
        # assert len(self.x[idx]) == len(self.y[idx]) + 2
        # return self.x[idx], self.y[idx]
        return [self.data[idx][0], self.data[idx][1]]

    def preprocess(self, x, y):
        data = []
        sentences = []
        for line in x:
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append(((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs), line))
            """
            line = ''.join(line)
            words = self.tokenizer.encode(line)
            sentences.append(words)
            i = i + 1
            """
        # print("sum_error: ", error_sum)
        # return sentences
        for sentence, label in zip(sentences, y):
            data.append((sentence, label))
        return data
    

    @staticmethod
    def collate_fn(batch):
        """
        train_data.sort(key=lambda data: len(data[0]), reverse=True)
        data_length = [len(data[0]) for data in train_data]
        data_x = [torch.LongTensor(data[0]) for data in train_data]
        data_y = [torch.LongTensor(data[1]) for data in train_data]
        mask = [torch.ones(l, dtype=torch.uint8) for l in data_length]
        data_x = pad_sequence(data_x, batch_first=True, padding_value=0)
        data_y = pad_sequence(data_y, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        return data_x, data_y, mask, data_length
        """
        sentences = [x[0][0] for x in batch]
        ori_sents = [x[0][1] for x in batch]
        labels = [x[1] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        data_length = [len(data) for data in ori_sents]
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0
        word_pad_idx = 0
        label_pad_idx = -1

        # padding data 初始化
        batch_data = word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(np.array(batch_data), dtype=torch.long)
        batch_label_starts = torch.tensor(np.array(batch_label_starts), dtype=torch.long)
        batch_labels = torch.tensor(np.array(batch_labels), dtype=torch.long)
        return [batch_data, batch_label_starts, batch_labels, ori_sents, data_length]


if __name__ == '__main__':
    # test
    # with open('/content/word_segment/data/datasave.pkl', 'rb') as inp:
    with open('./word_segment/data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    train_dataset = Sentence(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=train_dataset.collate_fn)

    for batch_data, batch_label_starts, batch_labels, ori_sents, length in train_dataloader:
        # print(input, label)
        print("batch_data_shape:", batch_data.shape)
        print("batch_label_starts_shape: ", batch_label_starts.shape)
        print("batch_labels: ", batch_labels.shape)
        print("length: ", length)
        break