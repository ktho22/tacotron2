from multiprocessing import Process, Queue, Pool
import os, glob, re, librosa, argparse, torch, pickle, multiprocessing, tqdm
import pickle as pkl
from os.path import *
import numpy as np

def decompose_hangul(text):
    Start_Code, ChoSung, JungSung = 44032, 588, 28
    ChoSung_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']
    JongSung_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    line_dec = ""
    line = list(text.strip())

    for keyword in line:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - Start_Code
            char1 = int(char_code / ChoSung)
            line_dec += ChoSung_LIST[char1]
            char2 = int((char_code - (ChoSung * char1)) / JungSung)
            line_dec += JungSung_LIST[char2]
            char3 = int((char_code - (ChoSung * char1) - (JungSung * char2)))
            line_dec += JongSung_LIST[char3]
        else:
            line_dec += keyword
    return line_dec


def preprocess_text(txtlist, dir_bin, lang):
    vocab_set = set()
    text_file_list = []
    dirts = ['`', '#', '@', '\|', '\ufeff']
    for fname in txtlist:
        gname = join(dir_bin, basename(fname))
        text_file_list.append(gname)
        print(fname)
        with open(fname, 'r') as f:
        #with open(fname, 'r') as f, open(gname, 'w') as g:
            line = f.readline()
            line = ''.join([xx for xx in line if xx not in dirts])
            if lang == 'korean':
                line = decompose_hangul(line)
            elif lang == 'english':
                line = line.lower()
            #g.write(line)
            vocab_set = vocab_set.union(set(line))

    vocab_dict = dict(zip(vocab_set, range(len(vocab_set))))
    print('Final vocab: ', vocab_dict)
    #with open(dir_bin + '/_vocab_dict.pkl', 'wb') as g:
    #    pkl.dump(vocab_dict, g)
    return text_file_list

def preprocess_spec(wavlist, dir_bin, sample_rate=16000, nfft=2048, type_filter='both'):
    '''
        Preprocessing for mel- and lin- spectrogram
        Args: wavlist, dir_bin
        Return: speclist [(wavname, mel_length, mel_bin_length, lin_length, lin_bin_length)]
    '''
    print('Start writing %s spectrogram binary files' % type_filter)
    offset_m, offset_l = 0, 0
    p = Pool()
    map_return = p.map_async(inner_process_spec, wavlist)
    p.close(); p.join()
    return map_return.get()


def inner_process_spec(wavname, sample_rate=22050, isMono=True, type_filter='both', frame_len_inMS=50, 
        frame_shift_inMS=12.5, n_fft=2048, ref_level_db=20, min_level_db=-100, mel_dim=80, wav_limit=1500):

    try:
        write_path_mel = join(dir_bin, basename(wavname)[:-4] + '.mel')

        audio,_ = librosa.load(wavname, sr=sample_rate, mono=isMono)
        mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=mel_dim)

        # params for stft
        window_len = int(np.ceil(frame_len_inMS * sample_rate / 1000))
        hop_length = int(np.ceil(frame_shift_inMS * sample_rate / 1000))

        D = librosa.stft(audio, n_fft=n_fft, win_length=window_len, window='hann', hop_length=hop_length)
        spec = np.abs(D)

        # mel-scale spectrogram generation
        spec_mel = np.dot(mel_basis, spec)
        spec_mel = 20 * np.log10(np.maximum(1e-5, spec_mel))
        # linear spectrogram generation
        #spec_lin = 20 * np.log10(np.maximum(1e-5, spec)) - ref_level_db

        # normalize
        spec_mel = np.clip(-(spec_mel - min_level_db) / min_level_db, 0, 1)
        spec_mel = spec_mel.T
        #spec_lin = np.clip(-(spec_lin - min_level_db) / min_level_db, 0, 1)
        #spec_lin = spec_lin.T
        
        if len(spec_mel) > wav_limit:
            return None

        with open(write_path_mel, 'wb') as w_mel:
            bin_spec_mel = pickle.dumps({'mel':spec_mel}, protocol=pickle.HIGHEST_PROTOCOL)
            #bin_spec_mel = pickle.dumps({'mel':spec_mel,'lin':spec_lin}, protocol=pickle.HIGHEST_PROTOCOL)
            w_mel.write(bin_spec_mel)
    except Exception as e:
        print(e)

    return wavname 

def remove_mismatched_txt(jobs):
    for dir_spec, dir_text, dir_bin in jobs:
        print(dir_spec)
        os.makedirs(dir_bin, exist_ok=True)

        txtlist = sorted(glob.glob(join(dir_bin, '*.txt')))
        mellist = sorted(glob.glob(join(dir_bin, '*.mel')))

        txtlist_ = [basename(xx[:-4]) for xx in txtlist]
        mellist_ = [basename(xx[:-4]) for xx in mellist]

        print(f'Before: txtlist_: {len(txtlist_)}, mellist_: {len(mellist_)}')

        #outlist = list(set(txtlist_) - set(mellist_))
        outlist = list(set(mellist_) - set(txtlist_))
        print(outlist)
        mellist_ = [join(dir_bin, f'{xx}.mel') for xx in outlist]
        #txtlist_ = [join(dir_bin, f'{xx}.txt') for xx in outlist]
        #for txtname in txtlist_:
        #    print(txtname)
        #    os.system(f'rm {txtname}')

        for melname in mellist_:
            print(melname)
            os.system(f'rm {melname}')

        txtlist = sorted(glob.glob(join(dir_bin, '*.txt')))
        mellist = sorted(glob.glob(join(dir_bin, '*.mel')))

        txtlist_ = [basename(xx[:-4]) for xx in txtlist]
        mellist_ = [basename(xx[:-4]) for xx in mellist]

        print(f'After: txtlist_: {len(txtlist_)}, mellist_: {len(mellist_)}')

def add_mismatched_txt(jobs):
    for dir_spec, dir_text, dir_bin in jobs:
        print(dir_spec)
        os.makedirs(dir_bin, exist_ok=True)

        txtlist = sorted(glob.glob(join(dir_bin, '*.txt')))
        mellist = sorted(glob.glob(join(dir_bin, '*.mel')))

        txtlist_ = [basename(xx[:-4]) for xx in txtlist]
        mellist_ = [basename(xx[:-4]) for xx in mellist]

        print(f'Before: txtlist_: {len(txtlist_)}, mellist_: {len(mellist_)}')

        outlist = list(set(mellist_) - set(txtlist_))
        txtlist_ = [join(dir_text, f'{xx}.txt') for xx in outlist]
        preprocess_text(txtlist_, dir_bin, 'korean')

        txtlist = sorted(glob.glob(join(dir_bin, '*.txt')))
        mellist = sorted(glob.glob(join(dir_bin, '*.mel')))

        txtlist_ = [basename(xx[:-4]) for xx in txtlist]
        mellist_ = [basename(xx[:-4]) for xx in mellist]

        print(f'After: txtlist_: {len(txtlist_)}, mellist_: {len(mellist_)}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataset')
    parser.add_argument('--wav_dir', type=str, help='input wav folder')
    parser.add_argument('--txt_dir', type=str, help='input txt folder')
    parser.add_argument('--bin_dir', type=str, help='output bin folder')
    parser.add_argument('--lang', type=str, default='korean')

    args = parser.parse_args()
    print('[*] Preprocessing Database')

    dir_spec = args.wav_dir
    dir_text = args.txt_dir
    dir_bin = args.bin_dir

    jobs = [#('/data4/data/KETTS76/wav', '/data4/data/KETTS76/txt', '/data4/data/KETTS76/bin_22050'),
            #('/data4/data/KNTTS/wav', '/data4/data/KNTTS/txt', '/data4/data/KNTTS/bin_22050'),
            #('/data4/data/etri/wav22.05', '/data4/data/etri/txt', '/data4/data/etri/bin_22050'),
            #('/data4/data/KETTS/30m_wav', '/data4/data/KETTS/30m_txt', '/data4/data/KETTS/30m_bin_22050'),
            ('/data4/data/KETTS/30f_wav', '/data4/data/KETTS/30f_txt', '/data4/data/KETTS/30f_bin_22050')]

    #add_mismatched_txt(jobs)
    remove_mismatched_txt(jobs)

    #for dir_spec, dir_text, dir_bin in jobs:
    #    print(dir_spec)
    #    os.makedirs(dir_bin, exist_ok=True)

    #    txtlist = sorted(glob.glob(join(dir_text, '*.txt')))
    #    wavlist = sorted(glob.glob(join(dir_spec, '*.wav')))

    #    mellist = sorted(glob.glob(join(dir_bin, '*.mel')))

    #    wavlist_ = [basename(xx[:-4]) for xx in wavlist]
    #    mellist_ = [basename(xx[:-4]) for xx in mellist]

    #    print(f'wavlist_: {len(wavlist_)}, mellist_: {len(mellist_)}')

    #    outlist = list(set(wavlist_) - set(mellist_))

    #    wavlist_ = [join(dir_spec, f'{xx}.wav') for xx in outlist]

    #    txtlist_ = [join(dir_spec, f'{xx}.txt') for xx in outlist]
    #    for wavname in wavlist_:
    #        os.system(wavname)
    #    #speclist = preprocess_spec(wavlist_, dir_bin) 

    #    ##assert len(txtlist)==len(wavlist), "number of txt files and wav files should be same"
    #    #print(len(txtlist))

    #    #txtlist = []
    #    #speclist = preprocess_spec(wavlist_, dir_bin) 
    #    #print(len(speclist), len(wavlist))
    #    #for specfile in speclist:
    #    #    if specfile == None:
    #    #        continue
    #    #    txtlist.append(os.path.join(dir_text, splitext(basename(specfile))[0])+'.txt')
    #    #textlist = preprocess_text(txtlist, dir_bin, args.lang)

    #    for p in multiprocessing.active_children():
    #        p.terminate()
