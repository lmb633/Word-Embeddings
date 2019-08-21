from torch.utils.data import Dataset
import os
import jieba
import collections
from os.path import join
from config import folder, data_path
import numpy as np
import pickle
import random


class WordDataSet(Dataset):
    def __init__(self, mode, skip_window=2, pre_generate=True):
        self.data = build_dataset(pre_generate)
        self.pairs = generate_pairs(self.data['data'], skip_window)
        print('all data pairs: ', len(self.pairs))
        split = int(len(self.pairs) - len(self.pairs) / 10)
        print('train test split: ', split)
        if mode == 'train':
            self.pairs = self.pairs[0:split]
        else:
            self.pairs = self.pairs[split:]

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


def load_file(folder_path):
    paths = [join(dirpath, name)
             for dirpath, dirs, files in os.walk(folder_path)
             for name in files
             if not name.startswith('.')]
    concat = u''
    for path in paths:
        with open(path, 'r', encoding='utf-8') as myfile:
            # print(path)
            content = myfile.read()
            concat += cleanse(content)

    return concat


def read_data(concat_word):
    """Extract as a list of words"""
    seg_list = jieba.cut(concat_word, cut_all=False)
    # print "/".join(seg_list)
    words = []
    for t in seg_list:
        words.append(t)

    return words


def build_dataset(pre_generate):
    if os.path.exists(data_path) and pre_generate:
        with open(data_path, 'rb') as file:
            datas = pickle.load(file)
            return datas
    concat = load_file(folder)
    print('All txt length: ', len(concat))
    words = read_data(concat)
    print('Word size: ', len(words))
    vocab = set(words)
    vocab_size = len(vocab)
    print('Vocabulary size %d' % vocab_size)

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    datas = {}
    datas['data'] = data
    datas['ount'] = count
    datas['dictionary'] = dictionary
    datas['reverse_dictionary'] = reverse_dictionary
    datas['vocab_size'] = vocab_size
    with open(data_path, 'wb') as file:
        pickle.dump(datas, file)
    return datas


def cleanse(content):
    content = content.replace('\n', '')
    content = content.replace('\r', '')
    content = content.replace('\u3000', '')
    return content


def generate_pairs(data, skip_window=2):
    pairs = []
    for i in range(skip_window + 1, len(data) - skip_window):
        for j in range(i - skip_window, i + skip_window + 1):
            if i != j:
                pairs.append([data[i], data[j]])
    return pairs


def get_neg_v_sampling(data, pos_pairs, count):
    neg_v = np.random.choice(
        data, size=(len(pos_pairs), count)).tolist()
    return neg_v


if __name__ == '__main__':
    dataset = WordDataSet('train', 2, False)
    reverse_dictionary = dataset.data['reverse_dictionary']
    for i in range(100):
        print(reverse_dictionary[dataset[i][0]], reverse_dictionary[dataset[i][1]])
