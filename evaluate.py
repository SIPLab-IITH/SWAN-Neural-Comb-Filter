import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
import argparse
import numpy as np
import tensorflow as tf
import scipy
import pysepm
from scipy.io import wavfile
from scipy import signal
from attLayers import MultiHeadAttn, MultiHeadAttnCausal
from utils import ceploss, LossLayer, dummy_loss, stft_mag, log

parser = argparse.ArgumentParser(
        'enhance',
        description="Neural Comb Filtering using SWAN - generate enhancements")
parser.add_argument("--model", type=str, help="Trained Model File")
parser.add_argument("--noisy_dir", type=str, help="Noisy input files")
parser.add_argument("--ref_dir", type=str, help="Reference files")
parser.add_argument("--out_dir", default='./', type=str, help="directory putting enhanced wav files")
parser.add_argument("--type", default='non-causal', type=str, help="Type of the model, either Causal or Non-Causal")

def evaluate(args):

    winLen=512
    hopLen=256
    fftLen=512
    
    if args.type=='causal':
        model=tf.keras.models.load_model(args.model,custom_objects={'ceploss':ceploss,'MHAttn':MultiHeadAttnCausal,'LossLayer':LossLayer,'dummy_loss':dummy_loss,'log':log,'stft_mag':stft_mag})    
    else:
        model=tf.keras.models.load_model(args.model,custom_objects={'ceploss':ceploss,'MHAttn':MultiHeadAttn,'LossLayer':LossLayer,'dummy_loss':dummy_loss,'log':log,'stft_mag':stft_mag})
    
    model=tf.keras.Model(inputs=model.input[0], outputs=model.layers[-3].output)
    
    noise_dir=args.noisy_dir
    clean_dir=args.ref_dir

    noise_names = [os.path.join(noise_dir,na) for na in os.listdir(noise_dir)
                if na.lower().endswith(".wav")]
    clean_names = [os.path.join(clean_dir,na) for na in os.listdir(noise_dir)
                if na.lower().endswith(".wav")]
    N = len(clean_names)
    
    window_fn = tf.signal.inverse_stft_window_fn(hopLen)
    
    pesq=[];stoi=[];csig=[];cbak=[];covl=[];ssnr=[]
    
    for clean_file, noise_file in zip(clean_names[:N], noise_names[:N]): 
        print('Processing', noise_file)
        fs,mixed_wav=wavfile.read(noise_file)
        mixed_wav=tf.pad(mixed_wav*1., [[256,256]], 'CONSTANT')
        mixed_spec=tf.signal.stft(mixed_wav, frame_length=winLen, frame_step=hopLen, fft_length=fftLen)
        mixed_wav=tf.reshape(mixed_wav, (1, -1))

        mask = tf.squeeze(model.predict(mixed_wav))

        enh_wav=tf.signal.inverse_stft(mixed_spec*tf.cast(mask, tf.complex128),
            frame_length=winLen, frame_step=hopLen, window_fn=window_fn).numpy()
        enh_wav=enh_wav[256:]
        
        fs, clean_wav = wavfile.read(clean_file)
        clean_wav = clean_wav*1.
        L=np.min([clean_wav.shape[0], enh_wav.shape[0]])
        clean_wav=clean_wav[:L]
        enh_wav=enh_wav[:L]
        pq,si,cs,cb,co,ss = evaluate_metrics(clean_wav, enh_wav,fs)
        pesq.append(pq);stoi.append(si);csig.append(cs);cbak.append(cb);covl.append(co);ssnr.append(ss)
        wavfile.write(args.out_dir+noise_file.split('/')[-1], fs, enh_wav/max(enh_wav))

    return np.mean(pesq),np.mean(stoi),np.mean(csig),np.mean(cbak),np.mean(covl),np.mean(ssnr)

def evaluate_metrics(speech_audio, noise_audio, fs):
    _,pesq = pysepm.pesq(speech_audio, noise_audio,fs)
    stoi = pysepm.stoi(speech_audio, noise_audio, fs)
    csig, cbak, covl = pysepm.composite(speech_audio, noise_audio, fs)
    ssnr = pysepm.SNRseg(speech_audio, noise_audio, fs)
    return pesq, stoi, csig, cbak, covl, ssnr

if __name__ == '__main__':
    args = parser.parse_args()
    pesq, stoi, csig, cbak, covl, ssnr = evaluate(args)
    print('\nPESQ:{:.4f}\nSTOI:{:.4f}\nCSIG:{:.4f}\nCBAK:{:.4f}\nCOVL:{:.4f}\nSSNR:{:.4f}\n'
              .format(pesq, stoi, csig, cbak, covl, ssnr))
