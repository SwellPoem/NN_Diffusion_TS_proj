## NN_Diffusion_TS_proj
The repository is a re-implementation of the paper 'Diffusion-TS: interpretable diffusion for general time series generation'.

### Data preparation
Three important folders can be obtained from the [Google Drive](https://drive.google.com/drive/folders/1YlfhfQs5-iQzQy4fS0zq-mzH9I_IMTWW?usp=sharing) folder:
- download <b>dataset.zip</b>, then unzip and copy it in the folder ```./Data```.
- download <b>energy.zip</b>, then unzip and copy it in the folder ```./output```.
- download <b>TimeGAN_output.zip</b>, then unzip and copy it in the repo folder. 

### Run the code
The project can be run by the notebook ```main.ipynb``` present in the repo. In the notebook are presented 3 experiment in order to show the performance and the results of the implemented work.

### Environment
The libraries needed for the proper functioning of the repo are provided as a ```requirements.txt```. 
The code have been tested in a virtual envoronment with Python 3.9.19.

### Directory structure
The directory structure comprises also the folders from Google Drive.
``` bash
NN_Diffusion_TS_proj/
│
├── Checkpoints_energy_24/
├── Checkpoints_sin_96/
├── Checkpoints_stock_24/
├── Checkpoints_stock_48/
│
├── Config/
│
├── Data/
│   └── datasets/
│
├── interpretability/
│   └── sines/
│       └── figures_1/
│
├── output/
│   ├── energy/
│   ├── stocks_forecasting/
│   └── stocks_goog/
│
├── Scripts/
│   ├── Data_utils/
│   │   └── real_datasets.py
│   ├── Metrics/
│   │   ├── cross_correlation.py
│   │   └── FID_calc.py
│   ├── decoder.py
│   ├── diffusion_TS.py
│   ├── display.py
│   ├── encoder.py
│   ├── get_dataLoader.py
│   ├── interpr_plots.py
│   ├── masking_utils.py
│   ├── model_classes.py
│   ├── optimizer.py
│   ├── sines.py
│   ├── transformer_model.py
│   ├── trainer.py
│   ├── ts2vec/
│   │   ├── Models/
│   │   │   ├── dilated_conv.py
│   │   │   ├── encoder_ts2vec.py
│   │   │   ├── losses.py
│   │   │   └── ts2vec.py
│   │   ├── utils.py
│   │   └── utility_func.py
│
├── TimeGAN_output/
│   ├── generated_data_energy.npy
│   ├── generated_data_goog.npy
│   ├── ori_data_energy.npy
│   └── ori_data_goog.npy
│
├── main.ipynb
├── requirements.txt
└──README.md
```

### Aknowledgements
Diffusion-TS: Interpretable Diffusion for General Time Series Generation: https://github.com/Y-debug-sys/Diffusion-TS
Codebase for "Time-series Generative Adversarial Networks (TimeGAN)": https://github.com/jsyoon0823/TimeGAN

