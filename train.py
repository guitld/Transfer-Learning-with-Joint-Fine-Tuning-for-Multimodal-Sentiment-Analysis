import os
import wandb
import torch
import argparse
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from data import MVSADataset, HatefulMemes
from pytorch_lightning.loggers import WandbLogger
from model import SentimentClassifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

parser = argparse.ArgumentParser(description='Sentiment Analysis Classifier')
parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
parser.add_argument('--classes', '-c', type=int, help='number of classes')
parser.add_argument('--dataset', '-d', help='select dataset for training')
parser.add_argument('--wandb', '-w', help='wandb\'s project name')
parser.add_argument('--epochs', '-e', type=int, default=30, help='number of epochs')
parser.add_argument('--batch', '-b', type=int, default=32, help='batch size')
parser.add_argument('--workers', '-nw', type=int, default=16, help='num_workers for loader')
parser.add_argument('--accelerator', '-a', default='gpu', help='choose accelerator for trainer')

args = parser.parse_args()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if args.dataset == 'hateful_memes':
    dataset = HatefulMemes()
elif args.dataset == 'mvsa':
    dataset = MVSADataset()

kfold = KFold(n_splits=10, random_state=42, shuffle=True)

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    print(f"------------- Fold {fold} -----------------")
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_ld = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=train_sampler, drop_last=True, num_workers=args.workers)
    test_ld = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=test_sampler, drop_last=True, num_workers=args.workers)
    
    model = SentimentClassifier(lr=args.lr, n_classes=args.classes)
    logger = WandbLogger(project=args.wandb, name=f'{args.wandb}-{fold}')
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, callbacks=[EarlyStopping(monitor='val_loss', patience=2)], logger=logger)

    trainer.fit(model, train_dataloaders=train_ld, val_dataloaders=test_ld)
    wandb.finish()
    torch.cuda.empty_cache()