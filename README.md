# NDVI forecasting


## Prerequisites
Create the `ndviforecasting` environment with the following dependencies:
```
conda create -n ndviforecasting python==3.8.8
conda activate ndviforecasting
pip install git+https://github.com/catalyst-team/catalyst.git
conda install pandas matplotlib scikit-learn wandb --channel conda-forge
```
And voila!

Then, clone the git repository:
```
https://github.com/waldnerf/NDVI_forecasting.git
cd NDVI_forecasting
```

If you want to save the performance of your models on [wandb](https://wandb.ai/site), you will need to link your account by either doing:
```
>> wandb login
```
or 
```
python
import wandb
wandb.login()
```
and follow the instructions.

## Content

Below is the architecture of the folder. Code in `run_model.py` is used to train a model, and `evaluate_model.py` serves for model evaluation. The different models that are available can be found in `nn/models`

```
.
├── README.md
├── run_model.py
├── run_model.ipynb
├── evaluate_model.py
├── data
├── nn
│   ├── loss
│   └── models
└── utils
```

## Walkthrough

