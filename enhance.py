import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
import argparse
import numpy as np
import tensorflow as tf
import scipy
from scipy.io import wavfile
from scipy import signal
from attLayers import MultiHeadAttn, MultiHeadAttnCausal
from utils import ceploss, LossLayer, dummy_loss, stft_mag, log

parser = argparse.ArgumentParser(
        'enhance',
        description="Neural Comb Filtering using SWAN - generate enhancements")
parser.add_argument("--model", type=str, help="Trained Model File")
parser.add_argument("--noisy_file", type=str, help="Noisy input file")
parser.add_argument("--out_dir", default='./', type=str, help="directory putting enhanced wav files")
parser.add_argument("--type", default='non-causal', type=str, help="Type of the model, either Causal or Non-Causal")

def enhance(args):

    winLen=512
    hopLen=256
    fftLen=512

    if args.type=='causal':
        model=tf.keras.models.load_model(args.model,custom_objects={'ceploss':ceploss,'MHAttn':MultiHeadAttnCausal,'LossLayer':LossLayer,'dummy_loss':dummy_loss,'log':log,'stft_mag':stft_mag})    
    else:
        model=tf.keras.models.load_model(args.model,custom_objects={'ceploss':ceploss,'MHAttn':MultiHeadAttn,'LossLayer':LossLayer,'dummy_loss':dummy_loss,'log':log,'stft_mag':stft_mag})
    model=tf.keras.Model(inputs=model.input[0], outputs=model.layers[-3].output)
    
    window_fn = tf.signal.inverse_stft_window_fn(hopLen)
    
    fs,mixed_wav=wavfile.read(args.noisy_file)
    mixed_wav=tf.pad(mixed_wav*1., [[256,256]], 'CONSTANT')
    mixed_spec=tf.signal.stft(mixed_wav, frame_length=winLen, frame_step=hopLen, fft_length=fftLen)
    mixed_wav=tf.reshape(mixed_wav, (1, -1))
    
    mask = tf.squeeze(model.predict(mixed_wav))

    enh_wav=tf.signal.inverse_stft(mixed_spec*tf.cast(mask, tf.complex128),
            frame_length=winLen, frame_step=hopLen, window_fn=window_fn).numpy()
    enh_wav=enh_wav[256:]
    
    wavfile.write(args.out_dir+'enhanced.wav', fs, enh_wav/max(enh_wav))

if __name__ == '__main__':
    args = parser.parse_args()
    enhance(args)
