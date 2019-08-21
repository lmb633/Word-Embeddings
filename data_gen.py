from torch.utils.data import Dataset
import os
import jieba
import collections
from os.path import join
from config import folder
import numpy as np
import random


class WordDataSet(Dataset):
    def __init__(self, mode, skip_window=2):
        concat = load_file(folder)
        print('All txt length: ', len(concat))
        word = read_data(concat)
        print('Word size: ', len(word))
        self.data, count, dictionary, reverse_dictionary, vocab_size = build_dataset(word)
        self.pairs = generate_pairs(self.data, skip_window)
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


def build_dataset(words):
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
    return data, count, dictionary, reverse_dictionary, vocab_size + 1


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
    dataset = WordDataSet('train', 2)
    for i in range(100):
        print(dataset[i][0])
