import pickle
import logging
import argparse
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import CWS
from dataloader import Sentence
from numba import cuda

from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    # print("Initial GPU Usage")
    # gpu_usage()                             

    torch.cuda.empty_cache()

    # print("GPU Usage after emptying the cache")
    # gpu_usage()


def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str, default='./pretrained_bert_models/bert-base-chinese/')
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--cuda', action='store_true', default=True)
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    start, end = -1, -1
    for j in range(len(x)):
        if id2tag[y[j]] == 'B':
            start = cur + j
        elif id2tag[y[j]] == 'M' and start != -1:
            continue
        elif id2tag[y[j]] == 'E' and start != -1:
            end = cur + j
            entities.add((start, end))
            start, end = -1, -1
        elif id2tag[y[j]] == 'S':
            entities.add((cur + j, cur + j))
            start, end = -1, -1
        else:
            start, end = -1, -1


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    print("cuda_available: ", use_cuda)

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim, args.bert_path)
    if use_cuda:
        model = model.cuda()
    print()
    print("type.named_parameters(): ", type(model.named_parameters()))
    print()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    test_data = DataLoader(
        dataset=Sentence(x_test[:1000], y_test[:1000]),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        train_loss = 0
        for idx, batch_samples in enumerate(tqdm(train_data)):
            batch_data, batch_token_starts, batch_tags, _, _ = batch_samples
            # assert batch_data.shape[1] < 512
            # print("batch_data: ", batch_data.shape)
            # print("batch_token_starts: ", batch_token_starts.shape)
            # print("batch_tags: ", batch_tags.shape)
            if use_cuda:
                batch_data = batch_data.cuda()
                batch_token_starts = batch_token_starts.cuda()
                batch_tags = batch_tags.cuda()
                # label = label.cuda()
                # mask = mask.cuda()
            batch_masks = batch_data.gt(0)
            label_masks = batch_tags.gt(-1)

            # forward
            loss = model((batch_data, batch_token_starts), attention_mask = batch_masks, label_masks = label_masks, tags = batch_tags)
            # train_loss += loss.item()
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            torch.cuda.empty_cache()
            # cuda.select_device(0)
            # cuda.close()
            # cuda.select_device(0)

            if step % 100 == 0:
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []
                scheduler.step(loss)

        # save first, then test
        path_name = "./save/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, path_name)
        logging.info("model has been saved in  %s" % path_name)

        # test
        entity_predict = set()
        entity_label = set()
        with torch.no_grad():
            model.eval()
            cur = 0
            for idx, batch_samples in enumerate(tqdm(train_data)):
                batch_data, batch_token_starts, batch_tags, _, length = batch_samples
                if use_cuda:
                    batch_data = batch_data.cuda()
                    batch_token_starts = batch_token_starts.cuda()
                    batch_tags = batch_tags.cuda()
                    # label = label.cuda()
                    # mask = mask.cuda()
                batch_masks = batch_data.gt(0)
                label_masks = batch_tags.gt(-1)
                predict = model.infer((batch_data, batch_token_starts), batch_masks, label_masks)

                for i in range(len(length)):
                    entity_split(batch_data[i, :length[i]], predict[i], id2tag, entity_predict, cur)
                    entity_split(batch_data[i, :length[i]], batch_tags[i, :length[i]], id2tag, entity_label, cur)
                    cur += length[i]

            right_predict = [i for i in entity_predict if i in entity_label]
            if len(right_predict) != 0:
                precision = float(len(right_predict)) / len(entity_predict)
                recall = float(len(right_predict)) / len(entity_label)
                logging.info("precision: %f" % precision)
                logging.info("recall: %f" % recall)
                logging.info("fscore: %f" % ((2 * precision * recall) / (precision + recall)))
            else:
                logging.info("precision: 0")
                logging.info("recall: 0")
                logging.info("fscore: 0")
            model.train()


if __name__ == '__main__':
    set_logger()
    main(get_param())
