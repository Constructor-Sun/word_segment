import codecs
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import pickle

"""
BERT_PATH = '/content/word_segment/pretrained_bert_models/bert-base-chinese'
INPUT_DATA = "/content/word_segment/data/train.txt"
SAVE_PATH = "/content/word_segment/data//datasave.pkl"
"""
BERT_PATH = './word_segment/pretrained_bert_models/bert-base-chinese'
INPUT_DATA = "./word_segment/data/train.txt"
SAVE_PATH = "./word_segment/data//datasave.pkl"
id2tag = ['B', 'M', 'E', 'S']  # B：分词头部 M：分词词中 E：分词词尾 S：独立成词
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []

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

"""
def handle_data():
    '''[CLS]
    :return:
    '''

    x_data = []
    y_data = []
    wordnum = 0
    line_num = 0
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line_num = line_num + 1
            line = line.strip()
            if not line:
                continue
            line_x = []
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                if (line[i] in id2word):
                    line_x.append(word2id[line[i]])
                else:
                    id2word.append(line[i])
                    word2id[line[i]] = wordnum
                    line_x.append(wordnum)
                    wordnum = wordnum + 1
            x_data.append(line_x)

            lineArr = line.split()
            line_y = []
            for item in lineArr:
                line_y.extend(getList(item))
            y_data.append(line_y)

    print(x_data[0])
    print([id2word[i] for i in x_data[0]])
    print(y_data[0])
    print([id2tag[i] for i in y_data[0]])
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=43)
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
"""

def cut_sentence(init_list, sublist_len, sep_word):
    """直接按最大长度切分"""
    list_groups = zip(*(iter(init_list),) * sublist_len)
    # end_list = [list(i) + list(sep_word) for i in list_groups]
    end_list = [list(i) for i in list_groups]
    count = len(init_list) % sublist_len
    if count != 0:
        end_list.append(init_list[-count:])
    else:
        end_list[-1] = end_list[-1][:-1]  # remove the last sep word
    return end_list

def handle_data():
    x_data = []
    y_data = []
    wordnum = 0
    i = 0
    cut_number = 0
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            words = []
            word_lens = []
            line = line.strip()
            lineArr = line.split()
            line = [item for item in line if item != " "]
            if (len(line) == 0 or len(line) < 0):
                print("line: ", line)
                continue
            """for token in line:
                if token not in id2word:
                    id2word.append(token)
                    word2id[token] = wordnum
                    wordnum += 1"""
            line_y = []
            for item in lineArr:
                line_y.extend(getList(item))
            
            if (len(line) != len(line_y)):
                print("!!!!!!!!!!!!!!!!!!!line: ", line)
                print("!!!!!!!!!!!!!!!!!!!line_y: ", line_y)

            if (len(line) > 512):
                sub_x_data = cut_sentence(line, 510, '@')
                sub_y_data = cut_sentence(line_y, 510, 'S')
                x_data.extend(sub_x_data)
                y_data.extend(sub_y_data)
                cut_number += 1
            else:
                x_data.append(line)
                y_data.append(line_y)
            

            i = i + 1
            if (i % 10000 == 0):
                print("finished processing:", i)

    print("total_sentence: ", i)
    print("total_sentence_cut_number: ", cut_number)
    print("x_data[0]: ", x_data[0])
    print("y_data[0]: ", y_data[0])
    # print("word2id: ", word2id)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=43)
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)

if __name__ == "__main__":
    handle_data()

"""
“
”
‘
’
—
"""