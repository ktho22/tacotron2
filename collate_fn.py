import torch
import torch.utils.data as data
import random
import numpy as np

class collate_class:
    def __init__(self, contents_ratio=0.5, style_ratio=1):
        '''
        contents_ratio: higher means mel will be the contents
        style_ratio: higher means mel will be the style
        '''
        self.__dict__.update(locals())

    def __call__(self, data):
        keys = data[0].keys()

        assert 'contents_mel' in keys or 'txt' in keys

        if 'contents_mel' not in keys:
            contents_domain = 'txt'
        elif 'txt' not in keys:
            contents_domain = 'contents_mel'

        if random.random() < self.contents_ratio:
            contents_domain = 'contents_mel'
        else:
            contents_domain = 'txt'

        if random.random() < self.style_ratio:
            style_domain = 'style_mel'
        else:
            style_domain = 'latent'

        n_batch = len(data)
        #data.sort(key=lambda x: len(x[contents_domain]), reverse=True)

        # placeholder for targets
        target_mel_len = torch.tensor([len(x['target_mel']) for x in data])
        max_target_mel_len = max(target_mel_len)
        max_lin_len = max([len(x['target_lin']) for x in data])
        target_mel = torch.zeros(n_batch, max_target_mel_len, data[0]['target_mel'].shape[-1])
        target_lin = torch.zeros(n_batch, max_lin_len, data[0]['target_lin'].shape[-1])

        # placeholder for contents
        if 'contents_mel' in keys:
            contents_mel_len = torch.tensor([len(x['contents_mel']) for x in data])
            max_contents_mel_len = max(contents_mel_len)
            contents_mel = torch.zeros(n_batch, max_contents_mel_len, data[0]['contents_mel'].shape[-1])
        if 'txt' in keys:
            txt_len = torch.tensor([len(x['txt']) for x in data])
            max_txt_len = max(txt_len)
            txt = torch.zeros(n_batch, max_txt_len).long()


        # placeholder for style
        assert 'style_mel' in keys or 'style' in keys
        if 'style_mel' in keys:
            style_mel_len = torch.tensor([len(x['style_mel']) for x in data])
            max_style_mel_len = max(style_mel_len)
            style_mel = torch.zeros(n_batch, max_style_mel_len, data[0]['style_mel'].shape[-1])

        gender = torch.zeros(n_batch).long()
        age = torch.zeros(n_batch).long()
        emotion = torch.zeros(n_batch).long()
        spkr = torch.zeros(n_batch).long()
        spkemb = torch.zeros((n_batch, 256))
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()

        filename = []

        for ii, item in enumerate(data):
            target_mel[ii, :len(item['target_mel'])] = torch.tensor(item['target_mel'])
            target_lin[ii, :len(item['target_lin'])] = torch.tensor(item['target_lin'])
            gate_padded[i, targetmel.size(1)-1:] = 1

            if 'style_mel' in keys:
                style_mel[ii, :len(item['style_mel'])] = torch.tensor(item['style_mel'])
            if 'contents_mel' in keys:
                contents_mel[ii, :len(item['contents_mel'])] = torch.tensor(item['contents_mel'])
            txt[ii, :len(item['txt'])] = torch.tensor(item['txt']).long()

            gender[ii]  = item['style']['gender']
            age[ii]     = item['style']['age']
            emotion[ii] = item['style']['emotion']

            if 'spkr' in item['style'].keys():
                spkr[ii] = item['style']['spkr']

            if 'speaker' in item['style'].keys():
                spkemb[ii] = torch.tensor(item['style']['speaker'])

            filename.append(item['filename'])

        out_list = ['target_mel', 'target_lin', 'target_mel_len',
                    'gate_padded',
                    'contents_mel', 'contents_mel_len',
                    'txt', 'txt_len',
                    'style_mel', 'style_mel_len',
                    'gender', 'age', 'emotion', 'spkr', 'spkemb', 
                    'filename']

        if not contents_domain == 'txt':
            exclude_list = ['txt', 'txt_len']
        elif not contents_domain == 'contents_mel':
            exclude_list = ['contents_mel', 'contents_mel_len']

        if not style_domain == 'style_mel':
            exclude_list.extend(['style_mel', 'style_mel_len'])
        if not style_domain == 'latent':
            exclude_list.extend(['gender', 'age', 'emotion', 'spkemb'])

        out_list = [xx for xx in out_list if xx not in exclude_list]
        return_dict = {k:v for k, v in locals().items() if k in out_list}

        assert len(out_list) == len(return_dict)

        return return_dict 
