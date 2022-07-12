# Neural Comb Filtering using Sliding Window Attention Network for Speech Enhancement

### Requirements
The following libraries/packages are required

**Package Name    Recommended Version**
  python					    3.9.7
  numpy               1.19.5
  scipy               1.7.1
  tensorflow          2.4.1
  pysepm              0.1

## Denoising a noisy file
Use the following command to enhance a noisy wav file
```bash
python enhance.py --model=<a pretrained model> --type=<Causal or Non causal processing> --noisy_file=<Path to the noisy file> --out_dir=<Path to the directory to save enhanced file>
```
Note that the Causal model type is used for the model trained with causal context.
```
Usage: python enhance.py [--model <path to a pretrained model> | pre_trained_models/SWAN_1L.h5 | pre_trained_models/SWAN_3L.h5 | pre_trained_models/SWAN_6L.h5
                                                               | pre_trained_models/SWAN_Causal_1L.h5 |  pre_trained_models/SWAN_Causal_3L.h5 | pre_trained_models/SWAN_Causal_6L.h5]
                         [--type <Model Type> | causal | non-causal(default) ]
                         [--noisy_file <Path directing to noisy file>]
                         [--out_dir <Directory to place enhanced files (By default it is current working directory)>]
```

## Evaluating the model
Use the following command to evaluate the metrics [PESQ, STOI, CSIG, CBAK, COVL, SSNR] on pretrained models
```bash
python evaluate.py --model=<a pretrained model> --type=<Causal or Non causal processing> --noisy_dir=<Path to the noisy files> --ref_dir=<Path to the reference files> --out_dir=<Path to the directory to save enhanced files>
```
```
Usage: python enhance.py [--model <path to a pretrained model> | pre_trained_models/SWAN_1L.h5 | pre_trained_models/SWAN_3L.h5 | pre_trained_models/SWAN_6L.h5
                                                               | pre_trained_models/SWAN_Causal_1L.h5 |  pre_trained_models/SWAN_Causal_3L.h5 | pre_trained_models/SWAN_Causal_6L.h5]
                         [--type <Model Type> | causal | non-causal(default) ]
                         [--noisy_dir <Path directing to noisy files>]
                         [--ref_dir <Path directing to reference files>]
                         [--out_dir <Directory to place enhanced files (By default it is current working directory)>]
```

