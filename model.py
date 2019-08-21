import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class Word2vec(nn.Module):
    def __init__(self, word_size, emb_dimension):
        super(Word2vec, self).__init__()
        self.word_size = word_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(word_size, emb_dimension)
        self.v_embeddings = nn.Embedding(word_size, emb_dimension)
        self.init_emb()

    def init_emb(self):
        """Initialize embedding weight like word2vec.
        The u_embedding is a uniform distribution in [-0.5/em_size, 0.5/emb_size], and the elements of v_embedding are zeroes.
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        print('self.u_embeddings.weight.data.shape', self.u_embeddings.weight.data.shape)
        print('emb_u', emb_u.shape)
        emb_v = self.v_embeddings(pos_v)
        print('emb_v', emb_v.shape)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        print('neg_emb_v', neg_emb_v.shape)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score) + torch.sum(neg_score))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    a = torch.LongTensor([i for i in range(32)])
    b = torch.LongTensor([i for i in range(32)])
    c = torch.LongTensor([i for i in range(32)]).unsqueeze(dim=1)
    print(c.shape)
    model = Word2vec(128, 5)
    result = model(a, b, c)
    # print(a)
    # print(result)


    # print(torch.rand(2,(2)))
