import torch
import pickle
import numpy as np
from dataloader import Sentence
from torch.utils.data import DataLoader
from tqdm import tqdm

def getList(input_str):
    '''
    单个分词转换为tag序列
    :param input_str: 单个分词
    :return: tag序列
    '''
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append(tag2id['S'])
    elif len(input_str) == 2:
        outpout_str = [tag2id['B'], tag2id['E']]
    else:
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        outpout_str.append(tag2id['B'])
        outpout_str.extend(M_list)
        outpout_str.append(tag2id['E'])
    return outpout_str

if __name__ == '__main__':
    model = torch.load('save/model_0.964787.pkl', map_location=torch.device('cpu'))
    output = open('cws_result.txt', 'w', encoding="utf-8")
    OUTPUT_PATH = 'test_4_result.txt'
    batch_size = 1

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    """
    with open('./test_4.txt', 'r', encoding='utf-8') as f:
        for test in f:
            flag = False
            test = test.strip()

            x = torch.LongTensor(1, len(test))
            mask = torch.ones_like(x, dtype=torch.uint8)
            length = [len(test)]
            for i in range(len(test)):
                if test[i] in word2id:
                    x[0, i] = word2id[test[i]]
                else:
                    x[0, i] = len(word2id)

            predict = model.infer(x, mask, length)[0]
            for i in range(len(test)):
                print(test[i], end='', file=output)
                if id2tag[predict[i]] in ['E', 'S']:
                    print(' ', end='', file=output)
            print(file=output)
    """
    x_data = []
    y_data = []
    with open(OUTPUT_PATH, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line = line.strip()
            lineArr = line.split()
            line = [item for item in line if item != " "]
            if (len(line) == 0 or len(line) < 0):
                print("line: ", line)
                continue
            line_y = []
            for item in lineArr:
                line_y.extend(getList(item))

            x_data.append(line)
            y_data.append(line_y)

            target_data = DataLoader(
                dataset=Sentence(x_data, y_data),
                shuffle=False,
                batch_size=batch_size,
                collate_fn=Sentence.collate_fn,
                drop_last=False,
                num_workers=6
            )

            i = 0
            for batch_samples in tqdm(target_data):
                batch_data, batch_token_starts, batch_tags, _, length = batch_samples
                batch_masks = batch_data.gt(0)
                label_masks = batch_tags.gt(-1)
                predict = model.infer((batch_data, batch_token_starts), batch_masks, label_masks)

                for batch in range(batch_size):
                    for j in range(len(x_data[i+batch])):
                        print(x_data[i+batch][j], end='', file=output)
                        if id2tag[predict[batch][j]] in ['E', 'S']:
                            print(' ', end='', file=output)
                    print(file=output)
                i += batch_size
    