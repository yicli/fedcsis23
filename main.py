import logging
import os
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess.feature_loader import FeatureDataset, collate_logs
from model.conv_net import ConvNet2Blk, ConvNet3Blk

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    run_name = 'conv1d_spawn_count_local'
    train_writer = SummaryWriter(os.path.join('runs', run_name))
    batch_sz = 16
    n_channels = [14, 7, 3]
    dataset_name = 'train_local_scaled'
    features = [
        'csv', 'SYSCALL_exit_isNeg', 'CUSTOM_openSockets_count',
        'CUSTOM_openFiles_count', 'CUSTOM_libs_count', 'spawn_count'
    ]

    logging.info('Initialising model ...')
    model = ConvNet2Blk(n_channels, 'batch', residual=True)
    criterion = BCELoss()
    optimiser = Adam(model.parameters())
    train_writer.add_graph(model, torch.zeros(batch_sz, n_channels[0], 1500))
    model = model.to(device)

    logging.info('Initialising data loader ...')
    # features = [
    #     'csv', 'SYSCALL_syscall', 'PROCESS_comm', 'SYSCALL_exit_isNeg',
    #     'CUSTOM_openSockets_count', 'CUSTOM_openFiles_count', 'CUSTOM_libs_count',
    #     'PROCESS_uid', 'PROCESS_gid'
    # ]
    train_set = FeatureDataset(dataset_name, features)
    train_loader = DataLoader(
        train_set, batch_size=batch_sz, shuffle=True,
        collate_fn=collate_logs
    )

    logging.info('Started training ...')
    for e in range(100):
        loss, acc = train_one_epoch(model, optimiser, criterion, device, train_loader, e)
        train_writer.add_scalar('loss', loss, e)
        train_writer.add_scalar('accuracy', acc, e)


def train_one_epoch(model, optimiser, criterion, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_n_right = 0
    n_sample = 0

    for iter, (batch, labels) in enumerate(data_loader):
        batch = batch.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        y_hat = model(batch)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()
        batch_n_right = n_right(y_hat.detach(), labels.detach())
        epoch_n_right += batch_n_right
        n_sample += len(batch)
        batch_acc = batch_n_right / len(batch)

        if iter % 10 == 9:
            logging.info('[%i, %i] loss: %.3f acc: %.2f' % (epoch+1, iter+1, loss.item(), batch_acc))

    epoch_loss /= (iter + 1)
    epoch_acc = epoch_n_right / n_sample
    logging.info('[%i] epoch loss: %.3f epoch acc: %.2f' % (epoch+1, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def n_right(y_hat, labels):
    prediction = (y_hat > 0.5).int()
    n_right = (prediction == labels).sum()
    return n_right.item()


if __name__ == '__main__':
    train()
