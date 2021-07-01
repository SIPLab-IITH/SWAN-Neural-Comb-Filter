import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
import argparse
import numpy as np
import tensorflow as tf
import scipy
from scipy.io import wavfile
from scipy import signal
from attLayers import MultiHeadAttn, MultiHeadAttn_Causal
from spLosses import ceploss

parser = argparse.ArgumentParser(
        'enhance',
        description="Neural Comb Filtering using SWAN - generate enhancements")
parser.add_argument("--out_dir", default='./', type=str, help="directory putting enhanced wav files")
parser.add_argument("--model", type=str, help="Trained Model File")
parser.add_argument("--noisy_file", type=str, help="Noisy input file")
parser.add_argument("--type", default='non-causal', type=str, help="Type of the model, either Causal or Non-Causal")

def enhance(args):

    winLen=512
    hopLen=256
    fftLen=512

    if args.type=='causal':
        model=tf.keras.models.load_model(args.model,custom_objects={'ceploss':ceploss, 'MultiHeadAttn':MultiHeadAttn_Causal})    
    else:
        model=tf.keras.models.load_model(args.model,custom_objects={'ceploss':ceploss, 'MultiHeadAttn':MultiHeadAttn})

    model1=tf.keras.models.Model(model.inputs[0], model.outputs[0])
    
    window_fn = tf.signal.inverse_stft_window_fn(hopLen)
    
    fs,noise_wav=wavfile.read(args.noisy_file)
    noise_complex_x=tf.signal.stft(tf.cast(tf.pad(noise_wav, [[256,256]], 'CONSTANT'), tf.float32),frame_length=winLen, 
                                                  frame_step=hopLen, fft_length=fftLen, pad_end=False)
    X = tf.constant(1./256, dtype=tf.float32) * tf.math.abs(noise_complex_x)
    X = tf.expand_dims(X,axis=0)
    
    mask = tf.squeeze(model1.predict(X))

    enh_wav=tf.signal.inverse_stft(noise_complex_x*tf.cast(mask, tf.complex64),
            frame_length=winLen, frame_step=hopLen, window_fn=window_fn).numpy()
    enh_wav=enh_wav[256:]
    
    wavfile.write(args.out_dir+'enhanced.wav', fs, enh_wav/max(enh_wav))

if __name__ == '__main__':
    args = parser.parse_args()
    enhance(args)
