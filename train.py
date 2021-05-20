import os
import sys
import logging

from tqdm import tqdm

import torch
import torch.cuda
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import theconf
from theconf import Config as C

import trainer

def main():
    device = torch.device(type= 'cuda', index=max(0, int(os.environ.get('LOCAL_RANK', -1))))
    
    if flags.local_rank > 0:
        dist.init_process_group(backend=flags.dist_backend, init_method='env://', world_size=int(os.environ['WORLD_SIZE']))
        torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True

    model = trainer.model.create(C.get()['architecture'])
    if flags.local_rank > 0:
        model = DDP(model, device_ids=[flags.local_rank], output_device=flags.local_rank)
    model.to(device=device)

    dataloader, sampler = trainer.dataset.create(C.get()['dataset'], C.get()['crop'],
                                              int(os.environ.get('WORLD_SIZE', 1), int(os.environ.get('RANK', -1))))  # DataLoader
    optimizer = trainer.optimizer.create(C.get()['optimizer'])(model.parameters())

    


if __name__ == '__main__':
    parser = theconf.ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None, help='set seed (default:0xC0FFEE)')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m torch.distributed.launch\'.')
    flags = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', stream=sys.stderr)

    main(flags):