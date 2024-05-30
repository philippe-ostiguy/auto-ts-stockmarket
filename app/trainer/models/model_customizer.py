import torch
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import BasePointLoss, level_to_outputs,quantiles_to_outputs
from app.shared.config.config_utils import ConfigManager
from app.trainer.models.common import MetricCalculation
import logging
from typing import Optional, Union
import numpy as np


class CustomTFT(TFT):

    def __init__(self, config_manager : Optional [ConfigManager] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config_manager is None:
            config_manager = ConfigManager(file='app/trainer/config.yaml')
        self.daily_returns = torch.empty(0)
        self._lower_index, self._upper_index = config_manager.confidence_indexes
        self.best_risk_on_return = None
        self.best_epoch = None

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan
        windows = self._create_windows(batch, step="val")
        n_windows = len(windows["temporal"])
        y_idx = batch["y_idx"]

        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))
        batch_sizes = []
        self.daily_returns = torch.empty(0)

        if n_batches != 1:
            raise ValueError(f"n_batches should be equal to 1, now it's {n_batches}. Think of increasing the value of inference_windows_batch_size"
                             f"when initializing the model")
        for i in range(n_batches):
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._create_windows(batch, step="val", w_idxs=w_idxs)
            original_outsample_y = torch.clone(windows["temporal"][:, -self.h :, y_idx])
            windows = self._normalization(windows=windows, y_idx=y_idx)
            (
                insample_y,
                insample_mask,
                _,
                outsample_mask,
                hist_exog,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)
            windows_batch = dict(
                insample_y=insample_y,  # [Ws, L]
                insample_mask=insample_mask,  # [Ws, L]
                futr_exog=futr_exog,  # [Ws, L+H]
                hist_exog=hist_exog,  # [Ws, L]
                stat_exog=stat_exog,
            )

            output_batch = self(windows_batch)
            preds, _, _ = self._inv_normalization(
                y_hat=output_batch, temporal_cols=batch['temporal_cols'], y_idx=batch["y_idx"]
            )
            self._compute_return_batch(targets=original_outsample_y,preds=preds)
            batch_sizes.append(len(output_batch))
        
        valid_loss = self._get_final_val_loss()
        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")
        self.log("valid_loss", valid_loss, prog_bar=True, on_epoch=True)
        self.log("val_RiskReturn", valid_loss, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(valid_loss)
        return valid_loss


    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_RiskReturn',avg_loss)
        if self.best_risk_on_return is None or avg_loss <  self.best_risk_on_return:
            self.best_risk_on_return= avg_loss
            self.best_epoch = self.current_epoch
        if torch.isinf(-self.best_risk_on_return).any():
            raise ValueError(f"self.best_risk_on_return is infinite")

        logging.warning(
            f"Best Return on Risk so far: {-self.best_risk_on_return}, achieved at epoch: {self.best_epoch}")
        super().on_validation_epoch_end()

class RiskReturn(BasePointLoss):

    def __init__(
        self, level=[80, 90], quantiles=None, delta: float = 1.0, horizon_weight=None, config_manager : Optional [ConfigManager] = None,
    ):

        qs, output_names = level_to_outputs(level)
        qs = torch.Tensor(qs)

        if quantiles is not None:
            _, output_names = quantiles_to_outputs(quantiles)
            qs = torch.Tensor(quantiles)

        super(RiskReturn, self).__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=len(qs),
            output_names=output_names,
        )

        self.quantiles = torch.nn.Parameter(qs, requires_grad=False)
        self.delta = delta

        if config_manager is None:
            config_manager = ConfigManager(file='app/trainer/config.yaml')

        self._lower_index, self._upper_index = config_manager.confidence_indexes


    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):

        daily_returns = MetricCalculation.gather_daily_returns(y, y_hat, lower_index=self._lower_index, upper_index=self._upper_index)

        logging.warning(f'\nNb of trade in update(): {len(daily_returns)} out of {len(y)} possible')
        if daily_returns.numel()==0:
            return torch.tensor(100.0)
        metrics = MetricCalculation.get_risk_rewards_metrics(daily_returns)
        return_on_risk = metrics['return_on_risk']
        if return_on_risk == torch.tensor(0.0):
            return torch.tensor(100.0)
        logging.warning(
            f'Weighted return on risk in _get_final_val_loss : {return_on_risk}')
        return -return_on_risk
