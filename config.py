import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder = 'data/《刘慈欣作品全集》(v1.0)'
print_freq = 1000
data_path = 'data/data.pkl'
vocabulary_size = 60898
emb_size = 128

def parse_args():
    parser = argparse.ArgumentParser(description='train word2vec')
    parser.add_argument('--end-epoch', type=int, default=50, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.002, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')

    args = parser.parse_args()
    return args
