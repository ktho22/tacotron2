import numpy as np
from scipy.io.wavfile import read
import torch
import time
import os
from os.path import join, exists


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def set_savepath(message):
    today = time.strftime('%y%m%d%2M%2S%Z')
    savepath = join('result', '{}-{}'.format(today, message))
    if not exists(savepath):
        os.makedirs(savepath)
    elif message=='test':
        os.system("rm -rf {}/*".format(savepath))
    else:
        input("Path already exists, wish to continue?")
        os.system("rm -rf {}/*".format(savepath))
        os.system('cp -r *.py dataset {}'.format(savepath))
    return savepath

