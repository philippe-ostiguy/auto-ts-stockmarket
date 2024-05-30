from torch.optim.lr_scheduler import (
    OneCycleLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from neuralforecast.losses.pytorch import HuberMQLoss, RMSE,MQLoss

LR_SCHEDULER_MAPPING = {
    "OneCycleLR": OneCycleLR,
    "ExponentialLR": ExponentialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
}

PYTORCH_CALLBACKS = {
    'EarlyStopping' : EarlyStopping,
    'ModelCheckPoint' : ModelCheckpoint

}

LOSS = {
    'RMSE' : RMSE,
    'MQLoss': MQLoss,
    'HuberMQLoss' : HuberMQLoss,
}



