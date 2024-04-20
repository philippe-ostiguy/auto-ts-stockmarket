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
from app.trainer.models.models import HyperpametersOptimizer, ModelBuilder
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


from app.trainer.models.model_customizer import CustomAutoTFT
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT, NHITS
from neuralforecast.auto import AutoTFT
from neuralforecast.losses.pytorch import HuberMQLoss
from neuralforecast.utils import AirPassengersDF
from ray import tune


Y_df = AirPassengersDF
Y_df['y'] = Y_df['y'].div(Y_df['y'].shift(1)) - 1
Y_df = Y_df.iloc[1:]
Y_train_df = Y_df[Y_df.ds <= '1959-12-31']
Y_test_df = Y_df[Y_df.ds > '1959-12-31']
horizon = len(Y_test_df)

clear_directory_content('lightning_logs')
print(f"Input size : {horizon}")


default_config = {
        "input_size": tune.choice([8, 16,32, 64, 128]),
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "val_check_steps" :1,
        "max_steps": 10 ,
        "random_seed": tune.randint(1, 20),
    }


num_samples=5
models = [AutoTFT(config =default_config,
                    loss=HuberMQLoss(quantiles=[0.05,.4,0.5,.6,0.95]),
                    h = 1, num_samples=3)]

nf = NeuralForecast(models=models ,freq='M')
nf.fit(df=Y_train_df, val_size=50, use_init_models=True)

Y_hat_df = nf.predict().reset_index()
#best_model_path = self._model_checkpoint.best_model_path
#self._best_model = self._model_to_train.load_from_checkpoint(best_model_path)
#self._best_model = self._model_to_train.load_from_checkpoint('tempo/best_model.ckpt')

fig, ax = plt.subplots(1, 1, figsize=(20, 7))
Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')

plot_df[['y', 'TFT']].plot(ax=ax, linewidth=2)

ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()

t = 5

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
    if config_manager.config['common']['hyperparameters_optimization']['is_optimizing']:
        HyperpametersOptimizer(config_manager).run()

    ModelBuilder(config_manager).run()

    print('Program finished successfully')
    if not torch.cuda.is_available():
        play_music()
