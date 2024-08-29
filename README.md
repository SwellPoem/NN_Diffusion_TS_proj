# NN_Diffusion_TS_proj
The repository is a re-implementation of the paper 'Diffusion-TS: interpretable diffusion for general time series generation'.

### Directory structure
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
