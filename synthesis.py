import numpy as np
import torch

from layers import TacotronSTFT
from audio_processing import griffin_lim

def synthesis_griffin_lim(mel,hparams):
    taco_stft = TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length, 
    sampling_rate=hparams.sampling_rate)
    mel_decompress = taco_stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), 
               taco_stft.stft_fn, 60) 
    return waveform
       