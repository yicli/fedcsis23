import logging
import os
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess.feature_loader import FeatureDataset, collate_logs
from model.conv_net import ConvNet1Blk, ConvNet2Blk, ConvNet3Blk

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('Using device %s' % device)

def train():
    run_name = 'aug_ker502010'
    train_writer = SummaryWriter(os.path.join('runs', run_name))
    val_writer = SummaryWriter(os.path.join('runs', run_name + '_val'))
    batch_sz = 64
    n_channels = [6, 8, 4]
    train_set_name = 'train_aug'
    valid_set_name = 'valid_aug'
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
    train_set = FeatureDataset(train_set_name, features)
    train_loader = DataLoader(
        train_set, batch_size=batch_sz, shuffle=True,
        collate_fn=collate_logs
    )
    valid_set = FeatureDataset(valid_set_name, features)
    valid_loader = DataLoader(
        valid_set, batch_size=batch_sz, shuffle=True,
        collate_fn=collate_logs
    )

    logging.info('Started training ...')
    for e in range(100):
        loss, acc = train_one_epoch(model, optimiser, criterion, device, train_loader, e)
        train_writer.add_scalar('loss', loss, e)
        train_writer.add_scalar('accuracy', acc, e)

        if (e + 1) % 20 == 0:
            val_loss, val_acc = validate(model, criterion, device, valid_loader)
            val_writer.add_scalar('loss', val_loss, e)
            val_writer.add_scalar('accuracy', val_acc, e)
            checkpoint_path = os.path.join('runs', run_name, 'checkpoint.tar')
            torch.save({
                'epoch': e,
                'model_state': model.state_dict(),
                'opt_state': optimiser.state_dict(),
                'loss': loss
            }, checkpoint_path)


def train_one_epoch(model, optimiser, criterion, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_n_right = 0
    n_sample = 0

    for i, (batch, labels, _) in enumerate(data_loader):
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

        if (i + 1) % 100 == 0:
            logging.info('[%i, %i] loss: %.3f acc: %.2f' % (epoch+1, i+1, loss.item(), batch_acc))

    epoch_loss /= (i + 1)
    epoch_acc = epoch_n_right / n_sample
    logging.info('[%i] epoch loss: %.3f epoch acc: %.2f' % (epoch+1, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def validate(model, criterion, device, data_loader):
    model.eval()
    epoch_loss = 0
    epoch_n_right = 0
    n_sample = 0

    with torch.no_grad():
        for i, (batch, labels, _) in enumerate(data_loader):
            batch = batch.to(device)
            labels = labels.to(device)

            y_hat = model(batch)
            loss = criterion(y_hat, labels)

            epoch_loss += loss.item()
            batch_n_right = n_right(y_hat, labels)
            epoch_n_right += batch_n_right
            n_sample += len(batch)

    epoch_loss /= (i + 1)
    epoch_acc = epoch_n_right / n_sample
    logging.info('Valid loss: %.3f Valid acc: %.2f' % (epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def n_right(y_hat, labels):
    prediction = (y_hat > 0.5).int()
    n_right = (prediction == labels).sum()
    return n_right.item()


if __name__ == '__main__':
    train()
