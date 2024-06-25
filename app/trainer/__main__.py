# ###############################################################################
#
#  The MIT License (MIT)
#  Copyright (c) 2023 Philippe Ostiguy
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################
from app.shared.config.config_utils import InitProject, ConfigManager
from app.shared.data_processor import (
    DataForModelSelector,
    FutureCovariatesProcessor,
    DataProcessorHelper
)

from app.shared.utils import play_music, clear_directory_content
from app.shared.factory import DataSourceFactory
from app.trainer.models.models import ModelBuilder
import torch

from app.trainer.models.model_customizer import CustomTFT, RiskReturn
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT, NHITS
from neuralforecast.auto import AutoTFT
from neuralforecast.losses.pytorch import HuberMQLoss
from neuralforecast.utils import AirPassengersDF
from ray import tune
from optuna.trial import Trial
from optuna import samplers
# import os
# import shutil


from tensorboard.backend.event_processing import event_accumulator
import os

def _save_metrics_from_tensorboardflow():
    metrics_dict = {}

    os.makedirs(f'models/TFT/tensorboard', exist_ok=True)
    for event_file in os.listdir('lightning_logs/version_0'):
        if not event_file.startswith('events.out.tfevents'):
            continue
        full_path = os.path.join('lightning_logs/version_0', event_file)
        ea = event_accumulator.EventAccumulator(full_path)
        ea.Reload()

        for tag in ea.Tags()['scalars']:
            metrics_dict[tag] = ea.Scalars(tag)

    for metric, scalars in metrics_dict.items():
        plt.figure(figsize=(10, 5))

        if metric == 'train_loss_step':
            steps = [scalar.step for scalar in scalars]
        else:
            steps = list(range(len(scalars)))
        if metric == 'valid_loss' or metric == 'ptl/val_loss':
            values = [scalar.value for scalar in scalars]
            steps, values = zip(*[(step, value) for step, value in zip(steps, values) if value <= 10])
        else:
            values = [scalar.value for scalar in scalars]


        print(metric)
        plt.plot(steps, values, label=metric)
        plt.xlabel('Steps' if metric == 'train_loss_step' else 'Epoch')
        plt.ylabel('Value')
        plt.title(metric)
        plt.legend(loc='upper right')
        plt.savefig(f"models/TFT/tensorboard/{metric.replace('/', '_')}.png")
        plt.close()
_save_metrics_from_tensorboardflow()
t = 5


# nixtla_ckpt_name = None
# for filename in os.listdir('lightning_logs/saved_nixtla'):
#     if filename.endswith('.ckpt'):
#         nixtla_ckpt_name = filename
#         break
#
# best_model_ckpt = None
# for filename in os.listdir(f'lightning_logs/version_0/checkpoints'):
#     if filename.endswith('.ckpt'):
#         best_model_ckpt = filename
#         break
#
# old_file_path = os.path.join(f'lightning_logs/version_0/checkpoints', best_model_ckpt)
# new_file_path_in_checkpoints = os.path.join(f'lightning_logs/version_0/checkpoints', nixtla_ckpt_name)
# os.rename(old_file_path, new_file_path_in_checkpoints)
# shutil.move(new_file_path_in_checkpoints, f'./{nixtla_ckpt_name}')
# destination_path = os.path.join('lightning_logs/saved_nixtla', nixtla_ckpt_name)
# shutil.copyfile(new_file_path_in_checkpoints, destination_path)




# Y_df = AirPassengersDF
# column_type=  Y_df['ds'].dtype
# col  = Y_df['unique_id'].dtype
# Y_df['y'] = Y_df['y'].div(Y_df['y'].shift(1)) - 1
# Y_df = Y_df.iloc[1:]
# Y_train_df = Y_df[Y_df.ds <= '1959-12-31']
# Y_test_df = Y_df[Y_df.ds > '1959-12-31']

# clear_directory_content('lightning_logs')
# clear_directory_content(os.path.expanduser('~/ray_results'))
#
#
# default_config = {
#         "input_size": tune.choice([8, 16,32, 64, 128]),
#         "hidden_size": tune.choice([64, 128, 256]),
#         "n_head": tune.choice([4, 8]),
#         "learning_rate": tune.loguniform(1e-4, 1e-1),
#         "scaler_type": tune.choice([None, "robust", "standard"]),
#         "batch_size": tune.choice([32, 64, 128, 256]),
#         "windows_batch_size": tune.choice([128, 256, 512, 1024]),
#         "val_check_steps" :1,
#         "max_steps": 15,
#         "inference"
#         "random_seed": tune.randint(1, 20),
#     }
#
# def objective(trial):
#     config = {
#         "input_size": trial.suggest_categorical("input_size", [8, 16, 32, 64, 128]),
#         "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
#         "n_head": trial.suggest_categorical("n_head", [4, 8]),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
#         "scaler_type": trial.suggest_categorical("scaler_type", [None, "robust", "standard"]),
#         "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
#         "windows_batch_size": trial.suggest_categorical("windows_batch_size", [128, 256, 512, 1024]),
#         "val_check_steps": 1,
#         "max_steps": 15,
#         "valid_batch_size": 3000,
#         "inference_windows_batch_size" : 3000,
#         "random_seed": trial.suggest_int("random_seed", 1, 20),
#     }
#     return config
#
# num_samples=5
# models = [AutoTFT(backend="optuna", search_alg=samplers.TPESampler(seed=42),
#                     config=objective,
#                     loss=HuberMQLoss(quantiles=[0.05,.4,0.5,.6,0.95]),valid_loss=RiskReturn(quantiles=[0.05,.4,0.5,.6,0.95]),
#                     h = 1, num_samples=2)]
#
# nf = NeuralForecast(models=models ,freq='M')
# nf.fit(df=Y_train_df, val_size=50, use_init_models=True, local_scaler_type='standard')

# Y_hat_df = nf.predict().reset_index()
# #best_model_path = self._model_checkpoint.best_model_path
# #self._best_model = self._model_to_train.load_from_checkpoint(best_model_path)
# #self._best_model = self._model_to_train.load_from_checkpoint('tempo/best_model.ckpt')

# fig, ax = plt.subplots(1, 1, figsize=(20, 7))
# Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
# plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
#
# plot_df[['y', 'TFT']].plot(ax=ax, linewidth=2)
#
# ax.set_title('AirPassengers Forecast', fontsize=22)
# ax.set_ylabel('Monthly Passengers', fontsize=20)
# ax.set_xlabel('Timestamp [t]', fontsize=20)
# ax.legend(prop={'size': 15})
# ax.grid()



if __name__ == "__main__":
    config_manager = ConfigManager(file='app/trainer/config.yaml', clean_data_for_model=True)
    InitProject.create_common_path()
    InitProject.create_custom_path(file='app/trainer/config.yaml')

    data_processor_helper = DataProcessorHelper(config_manager=config_manager)
    sources = config_manager.get_sources()
    for source, is_input in config_manager.get_sources():
        data_for_source = config_manager.get_config_for_source(source, is_input)

        data_source = DataSourceFactory.implement_data_source(
            data_for_source,
            config_manager,
            data_processor_helper=data_processor_helper,
            is_input_feature=is_input,
        ).run()
    if config_manager.config["inputs"]["future_covariates"]["data"]:
        for window in range(config_manager.config['common']['cross_validation']['sliding_windows']):
            FutureCovariatesProcessor(
                config_manager, data_processor_helper, sliding_window=window
            ).run()
    if config_manager.config['common']['engineering']:
        DataForModelSelector(config_manager).run()

    ModelBuilder(config_manager).run()

    print('Program finished successfully')
    if not torch.cuda.is_available():
        play_music()
