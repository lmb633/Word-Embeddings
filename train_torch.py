from config import parse_args, device, print_freq, vocabulary_size, emb_size
from data_gen import WordDataSet, get_neg_v_sampling
import torch
from model import Word2vec, AverageMeter
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms

global data


def train_net(args):
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0
    train_set = WordDataSet('train')
    data = train_set.data
    valid_set = WordDataSet('valid')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    if args.checkpoint is None:
        print('train from begining')
        model = Word2vec(vocabulary_size, emb_size)
    else:
        print('load from checkpoint')
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    print('optimizer.lr ', optimizer.state_dict()['param_groups'][0]['lr'])

    model = model.to(device)
    for epoch in range(start_epoch, args.end_epoch):
        train_loss = train(train_loader, model, optimizer, epoch)

        valid_loss = valid(valid_loader, model)

        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print('Epochs since last improvement: ', epochs_since_improvement)
        else:
            epochs_since_improvement = 0

        save_checkpoint(epoch, epochs_since_improvement, model, best_loss, is_best, optimizer)
        print('optimizer.lr ', optimizer.state_dict()['param_groups'][0]['lr'])


def train(train_loader, model, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    for i, pairs in enumerate(train_loader):
        pos_u = [pair[0] for pair in pairs]
        pos_v = [pair[1] for pair in pairs]
        neg_v = get_neg_v_sampling(data, pos_u, 5)
        pos_u = Variable(torch.LongTensor(pos_u)).to(device)
        pos_v = Variable(torch.LongTensor(pos_v)).to(device)
        neg_v = Variable(torch.LongTensor(neg_v)).to(device)

        optimizer.zero_grad()
        loss = model(pos_u, pos_v, neg_v)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        # if i % print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]  Loss this batch:{3} (avg:{4})'.format(epoch, i, len(train_loader), losses.val, losses.avg))
    return losses.avg


def valid(valid_loader, model, epoch):
    model.eval()
    losses = AverageMeter()
    for i, pairs in enumerate(valid_loader):
        pos_u = [pair[0] for pair in pairs]
        pos_v = [pair[1] for pair in pairs]
        neg_v = get_neg_v_sampling(data, pos_u, 5)

        pos_u = Variable(torch.LongTensor(pos_u)).to(device)
        pos_v = Variable(torch.LongTensor(pos_v)).to(device)
        neg_v = Variable(torch.LongTensor(neg_v)).to(device)

        loss = model(pos_u, pos_v, neg_v)
        losses.update(loss.item())
    print('Validation: epoch: {0} Loss {1:.4f}'.format(epoch, losses.avg))
    return losses.avg


def save_checkpoint(epoch, epochs_since_improvement, model, best_loss, is_best, optimizer):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': best_loss,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')
    return


if __name__ == '__main__':
    args = parse_args()
    train_net(args)
