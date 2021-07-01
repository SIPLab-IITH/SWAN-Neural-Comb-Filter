import numpy as np
from amfm_decompy import pYAAPT, basic_tools
import librosa

## Segmenting SSNR (SSSNR) into voiced and unvoiced SSNR  

def get_f0(wav, fs):
    signal = basic_tools.SignalObj(data=wav/np.power(2,15), fs=fs)
    pitch=pYAAPT.yaapt(signal,**{'frame_length':32.0, 
        'frame_space':16.0,'tda_frame_length':32.0})
    return pitch.samp_values

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

def compute_snr(clean_speech_framed, processed_speech_framed):
    eps=np.finfo(np.float64).eps
    MIN_SNR     = -10 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1] # remove last frame -> not valid
    return np.mean(segmental_snr)

def SNRseg(clean_speech, processed_speech,fs, frameLen=0.032, overlap=0.50):
    
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    f0 = get_f0(clean_speech, fs)
    #print(f0.shape, clean_speech_framed.shape)
    L = min(f0.shape[0], clean_speech_framed.shape[0])
    f0 = f0[:L]
    clean_speech_framed = clean_speech_framed[:L]
    processed_speech_framed = processed_speech_framed[:L]
    #print(f0.shape, clean_speech_framed.shape)
    voiced_clean = clean_speech_framed[f0>0]
    voiced_processed = processed_speech_framed[f0>0]
    unvoiced_clean = clean_speech_framed[f0<=0]
    unvoiced_processed = processed_speech_framed[f0<=0]
    voiced_snr = compute_snr(voiced_clean, voiced_processed)
    unvoiced_snr = compute_snr(unvoiced_clean, unvoiced_processed)
    return voiced_snr, unvoiced_snr
    

