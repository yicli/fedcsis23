import logging
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess.feature_loader import FeatureDataset, collate_logs
from model.conv_net import ConvNet


def train_one_epoch(model, optimiser, criterion, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_n_right = 0
    n_sample = 0

    for iter, (batch, labels) in enumerate(data_loader):
        batch = batch.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        y_hat = model.forward(batch)
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
    import os
    import torch
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_name = 'conv_net_1d_local'
    train_writer = SummaryWriter(os.path.join('runs', run_name, 'train'))

    logging.info('Initialising data loader ...')
    # features = [
    #     'csv', 'SYSCALL_syscall', 'PROCESS_comm', 'SYSCALL_exit_isNeg',
    #     'CUSTOM_openSockets_count', 'CUSTOM_openFiles_count', 'CUSTOM_libs_count',
    #     'PROCESS_uid', 'PROCESS_gid'
    # ]
    features = [
        'csv', 'SYSCALL_exit_isNeg', 'CUSTOM_openSockets_count',
        'CUSTOM_openFiles_count', 'CUSTOM_libs_count'
    ]
    train_set = FeatureDataset('train_local_scaled', features)
    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True,
        collate_fn=collate_logs
    )

    logging.info('Initialising model ...')
    model = ConvNet(149, 50, 15, 5, 'batch', residual=True)
    criterion = BCELoss()
    optimiser = Adam(model.parameters())
    train_writer.add_graph(model, torch.zeros(16, 149, 1500))
    model = model.to(device)

    logging.info('Started training ...')
    for e in range(500):
        loss, acc = train_one_epoch(model, optimiser, criterion, device, train_loader, e)
        train_writer.add_scalar('loss', loss, e)
        train_writer.add_scalar('accuracy', acc, e)
