import warnings
import numpy as np
import torch
import os
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch import distributed as dist

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id", default=0)
args = parser.parse_args()

torch.distributed.init_process_group(backend="nccl", init_method='env://')
device = torch.device('cuda:{}'.format(args.local_rank))
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()

warnings.filterwarnings('ignore')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import MTL_hparam
from MTL_hparam import val_portion, training_epoch
from MTL_utils import tokenize_and_align_data
from MTL_dataloader import Rap_Dataset, grab_train_data
from MTL_model import Beat_Head, Duration_Head, Pitch_Head, Backbone
from MTL_funcs import Trainer

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    training_data = grab_train_data()

    tokenized_datasets = tokenize_and_align_data(training_data)

    dataset = Rap_Dataset(tokenized_datasets)

    # loading model
    total = int(len(training_data['tokens']))
    train_len = int(np.floor(total * (1 - val_portion)))
    val_len = int(total - train_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    # initialize
    global Backbone
    Backbone = Backbone().to(device)
    Beat_Head = Beat_Head().to(device)
    Duration_Head = Duration_Head().to(device)
    Pitch_Head = Pitch_Head().to(device)
    Backbone.init()
    Beat_Head.init()
    Duration_Head.init()
    Pitch_Head.init()

    Backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Backbone).to(device)
    Beat_Head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Beat_Head).to(device)
    Duration_Head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Duration_Head).to(device)
    Pitch_Head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Pitch_Head).to(device)

    Backbone = DDP(Backbone.to(device), device_ids=[args.local_rank], output_device=args.local_rank,
                   find_unused_parameters=True).to(device)
    Beat_Head = DDP(Beat_Head.to(device), device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=True).to(device)
    Duration_Head = DDP(Duration_Head.to(device), device_ids=[args.local_rank], output_device=args.local_rank,
                        find_unused_parameters=True).to(device)
    Pitch_Head = DDP(Pitch_Head.to(device), device_ids=[args.local_rank], output_device=args.local_rank,
                     find_unused_parameters=True).to(device)
    # Backbone = torch.nn.DataParallel(Backbone)
    # beat_model = torch.nn.DataParallel(beat_Head)
    # duration_model = torch.nn.DataParallel(duration_Head)
    # pitch_model = torch.nn.DataParallel(pitch_Head)

    trainer = Trainer(backbone=Backbone, beat_head=Beat_Head, duration_head=Duration_Head, pitch_head=Pitch_Head,
                      device=device, beat_criterion=torch.nn.CrossEntropyLoss(), duration_criterion=torch.nn.MSELoss(),
                      pitch_criterion=torch.nn.CrossEntropyLoss())

    with open(MTL_hparam.record_file, 'a') as f:
        f.write('script: {}\r\n'.format(sys.argv[0]))
        f.write('model: {}\r\n'.format(MTL_hparam.model_name))
        f.write('dataset: {}\r\n'.format(MTL_hparam.dataset_path))
        f.write('training_data_size: {}\r\n'.format(train_len))
        f.write('val_data_size: {}\r\n'.format(val_len))
        f.write('training_epoch: {}\r\n'.format(training_epoch))
        f.write('task_type: {}\r\n\r\n'.format(MTL_hparam.task))

    trainer.train(train_dataset, training_epoch, val_dataset)
    # trainer.validate(val_dataset)
