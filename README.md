# Neural Comb Filtering using Sliding Window Attention Network for Speech Enhancement

### Denoising a Noisy file
Use the following command to enhance a noisy wav file
```bash
python enhance.py --model=<a pretrained model> --type=<Causal or Non causal processing> --noisy_file=<Path to the noisy file> --out_dir=<Path to the directory to save enhanced file>
```
Note that the Causal model type is used for the model trained with causal context.
```
Usage: python enhance.py [--model <path to a pretrained model> | pre_trained_models/SWAN_1L.h5 | pre_trained_models/SWAN_3L.h5 
                               | pre_trained_models/SWAN_6L.h5 |  pre_trained_models/SWAN_Causal_1L.h5 | pre_trained_models/SWAN_Mel_Spectral_1L.h5 ]
                         [--type <Model Type> | causal | non-causal(default) ]
                         [--noisy_file <Path directing to noisy file>]
                         [--out_dir <Directory to place enhanced files (By default it is current working directory)>]
```
