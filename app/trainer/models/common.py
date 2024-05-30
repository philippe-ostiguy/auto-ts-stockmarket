import torch
from app.shared.config.config_utils import ConfigManager
from typing import Optional, Dict, Tuple
import os

class MetricCalculation:

    _metrics = {}
    def __int__(self):
        pass

    @classmethod
    def get_risk_rewards_metrics(cls,daily_returns : torch.Tensor,
                                 config_manager : Optional[ConfigManager] = None,
                                 is_checking_nb_trades : Optional[bool] = True) -> Dict:
        if config_manager is None:
            config_manager = ConfigManager(file='app/trainer/config.yaml')
        cls._metrics = {}
        cls._metrics["nb_of_trades"] = daily_returns.shape[0]
        if cls._metrics["nb_of_trades"] <= config_manager.config['common']['min_nb_trades'] and is_checking_nb_trades:
            return cls._set_zero_to_metrics(cls._metrics["nb_of_trades"])
        if cls._metrics["nb_of_trades"] == 0:
            return cls._set_zero_to_metrics(cls._metrics["nb_of_trades"])
        cls._metrics['annualized_return'] = torch.prod(1 + daily_returns) ** (
                252.0 / daily_returns.shape[0]) - 1
        cls._metrics['annualized_risk'] = daily_returns.std() * (252 ** 0.5)
        if cls._metrics['annualized_risk'].item() < 1e-6:
            cls._metrics['return_on_risk'] = torch.tensor(0.0, dtype=torch.float)
        else:
            cls._metrics['return_on_risk'] = cls._metrics['annualized_return'] / cls._metrics['annualized_risk']

        return cls._metrics


    @classmethod
    def _set_zero_to_metrics(cls, nb_of_trades) -> Dict:
        cls._metrics = {
            'annualized_return': torch.tensor(0.0),
            'annualized_risk': torch.tensor(0.0),
            'return_on_risk': torch.tensor(0.0),
            'nb_of_trades' : nb_of_trades
        }
        return cls._metrics



    @classmethod
    def gather_daily_returns(cls,
                             y : torch.Tensor,
                             y_hat : torch.Tensor,
                             lower_index :int,
                             upper_index : int) -> torch.Tensor:

        daily_returns = torch.empty(0)
        targets_size = len(y)
        if os.path.exists('tempo/returns.txt'):
            os.remove('positive_returns.txt')
        for item in range(targets_size):
            target, low_predictions, high_predictions = \
                cls.convert_torch_to_list(y=y,
                                          y_hat=y_hat,
                                          item=item,
                                          lower_index=lower_index,
                                          upper_index=upper_index)
            if low_predictions > 0 and high_predictions > 0:
                daily_returns = torch.cat((daily_returns, target), dim=0)
                with open('tempo/returns.txt', 'a') as f:
                    f.write(f"{target.item()}\n")
            elif high_predictions < 0 and low_predictions < 0:
                daily_returns = torch.cat((daily_returns, -target), dim=0)
                with open('tempo/returns.txt', 'a') as f:
                    f.write(f"{target.item()}\n")

        return daily_returns




    @staticmethod
    def convert_torch_to_list(y : torch.Tensor,
                              y_hat : torch.Tensor,
                              item : int, lower_index : int,
                              upper_index : int) -> Tuple[torch.Tensor,int,int]:
        target = y[item]
        values = y_hat[item].detach().cpu().numpy().squeeze()
        list_preds = values.tolist()
        low_predictions = list_preds[lower_index]
        high_predictions = list_preds[upper_index]
        return target, low_predictions,high_predictions
