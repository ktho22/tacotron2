import argparse, librosa
import numpy as np

import torch
import torch.nn as nn
import torchaudio

from model import Tacotron as Tacotron
from collate_fn import collate_fn
from dataset import *
from util import *
from griffin_lim import GLA
from audio_processing import *

parser = argparse.ArgumentParser(description='training script')
# data load
parser.add_argument('--data', type=str, default='vctk', help='vctk')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
# generation option
parser.add_argument('--out_dir', type=str, default='generated', help='')
parser.add_argument('--init_from', type=str, default='./model_545th.pt', help='load parameters from...')
parser.add_argument('--caption', type=str, default='', help='text to generate speech')
parser.add_argument('--speaker_id', type=str, default='0', help='speaker id to generate speech, seperate by comma for mixing id')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0, help='value between 0~1, use this for scheduled sampling')
# audio related option
parser.add_argument('--n_fft', type=int, default=2048, help='fft bin size')
parser.add_argument('--sample_rate', type=int, default=16000, help='sampling rate')
parser.add_argument('--frame_len_inMS', type=int, default=50, help='used to determine window size of fft')
parser.add_argument('--frame_shift_inMS', type=int, default=12.5, help='used to determine stride in sfft')
parser.add_argument('--num_recon_iters', type=int, default=50, help='# of iteration in griffin-lim recon')
# misc
parser.add_argument('--gpu', type=int, nargs='+', help='index of gpu machines to run')
parser.add_argument('--seed', type=int, default=0, help='random seed')
new_args = vars(parser.parse_args())

# load and override some arguments
checkpoint = torch.load(new_args['init_from'], map_location=lambda storage, loc: storage)
args = checkpoint['args']
for i in new_args:
    args.__dict__[i] = new_args[i]

torch.manual_seed(args.seed)

if args.gpu is None:
    args.use_gpu = False
    args.gpu = []
    device = torch.device('cpu')
    print("[*] Run in CPU mode")
else:
    args.use_gpu = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu[0])
    device = torch.device('cuda:{}'.format(args.gpu[0]))
    print("[*] Run in GPU mode")


model = Tacotron(args)
if args.init_from:
    model.load_state_dict(checkpoint['state_dict'])
    model.reset_decoder_states()
    print('loaded checkpoint %s' % (args.init_from))

model = model.eval()
GLA = Griffin_Lim(filter_length=args.n_fft)
if args.use_gpu:
    model = model.cuda()
    stft = stft.cuda()

def main():
    db = TTSDataset()
    loader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True)
    model_name = args.init_from.split('/')[-1][:-3]
    

    n_samples = 100
    for ii, sample in enumerate(loader):
       # for e in range(7):
        for k, v in sample.items():
            if k in ['filename', 'contents_domain']:
                continue
            sample[k] = v.to(device)

        wave, attentions = generate(sample)
        contents_filename = os.path.basename(sample['filename'][0]['input'])[:-4]
        style_filename = os.path.basename(sample['filename'][0]['ref'])[:-4]
        target_filename = os.path.basename(sample['filename'][0]['target'])[:-4]

        outpath1 = '%s/%s_%s_%s_%s.wav' % (args.out_dir, model_name, contents_filename, style_filename, target_filename)
        librosa.output.write_wav(outpath1, wave, 16000)
        outpath2 = '%s/%s_%s_%s_%s.png' % (args.out_dir, model_name, contents_filename, style_filename, target_filename)
        saveAttention(None, attentions, outpath2)
        if ii >= n_samples:
            break

    for ii, sample in enumerate(loader):
        print(ii)
        txt, mel, lin, txt_len, gender, age, emotion, emb, filename = sample
        if args.caption:

            for e in range(7):
                wave, attentions = generate(txt, mel, lin, txt_len, 
                        gender=gender, age=age, emotion=torch.tensor([e]))
                outpath1 = '%s/%s_e%d_%s.wav' % (args.out_dir, model_name, e, args.caption[:10])
                librosa.output.write_wav(outpath1, wave, 16000)
                outpath2 = '%s/%s_e%d_%s.png' % (args.out_dir, model_name, e, args.caption[:10])
                saveAttention(None, attentions, outpath2)
            break
        else:
            if ii >= n_samples:
                break
            waves, attentions = generate(txt, mel, lin, txt_len, gender=gender, age=age, emotion=emotion)
            for jj, (wave, attention) in enumerate(zip(waves, attentions)):
                outpath1 = '%s/%s_e%d_%d.wav' % (args.out_dir, model_name, emotion[0], jj)
                librosa.output.write_wav(outpath1, wave, 16000)
                outpath2 = '%s/%s_e%d_%d.png' % (args.out_dir, model_name, emotion[0], jj)
                saveAttention(None, attentions, outpath2)

def to_tensor(txt):
    txt = decompose_hangul(txt)
    txt = list(filter(None, [db.char2onehot(xx) for xx in txt]))
    txt = torch.tensor(txt).unsqueeze(0)
    txt_len[0] = txt.shape[-1]
    
        
def generate(sample):
    pred_mel, pred_lin = model(**sample)
    attentions = torch.cat(model.attn_weights, dim=-1)
    model.reset_decoder_states()

    magnitudes = denormalize_magnitudes(pred_lin.data)
    magnitudes = db_to_amplitude(magnitudes)
    magnitudes = magnitudes.permute(0, 2, 1) # Convert (N, L, C) to (N, C, L)
    signal = GLA.gla(magnitudes)

    return signal, attentions

if __name__ == '__main__':
    main()
