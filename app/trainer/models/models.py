import shutil
import inspect
import pandas as pd
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
import itertools
from typing import Optional
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from app.shared.config.config_utils import ConfigManager
from app.shared.utils import clear_directory_content, read_json, read_csv_to_pd_formatted, create_custom_trading_days
from app.shared.config.constants import DATASETS
from app.trainer.models.common import MetricCalculation
from .model_customizer import RiskReturn
import copy
import pickle
from abc import ABC
import torch
from neuralforecast.models import TFT
from neuralforecast import NeuralForecast
plt.ioff()


CUSTOM_MODEL = {'TFT' : TFT}


class BaseModelBuilder(ABC):
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None

    ):
        if config_manager is None:
            config_manager = ConfigManager(file='app/trainer/config.yaml')

        self._config_manager = config_manager
        self._config = self._config_manager.config
        self._params = {}
        self._datasets = DATASETS
        self._model_dir = ''
        self._lightning_logs_dir = ''
        self._logger = None
        self._model_name = ''
        self._lower_index, self._upper_index = config_manager.confidence_indexes
        self._best_model = ''
        self._model_to_train = ''
        self._values_retriever = ''


    def _assign_params(self, hyperparameters_phase : Optional[str] = 'hyperparameters'):
        params = {}
        model_config = read_json(
            "resources/configs/models_args.json"
        )["common"]
        for item in model_config.keys():
            if item in self._config[hyperparameters_phase]["common"]:
                params[item] = self._config[hyperparameters_phase]["common"][item]
        return params

    @staticmethod
    def _find_closest_value(lst, K, exclude):
        return min((abs(val - K), val) for val in lst if val not in exclude)[1]

    def _adjust_likelihood(self):
        if 'likelihood' in self._params and 'confidence_level' in self._params and \
                self._params['confidence_level'] != 0.5 and (
                self._params['confidence_level']
                not in self._params['likelihood'] or (
                        1 - self._params['confidence_level'])
                not in self._params['likelihood']):
            to_remove_1 = self._find_closest_value(self._params['likelihood'],
                                                   self._params[
                                                       'confidence_level'],
                                                   exclude=[0.5])
            self._params['likelihood'].remove(to_remove_1)

            to_remove_2 = self._find_closest_value(self._params['likelihood'],
                                                   1 - self._params[
                                                       'confidence_level'],
                                                   exclude=[0.5])
            self._params['likelihood'].remove(to_remove_2)
            self._params['likelihood'].append(self._params['confidence_level'])
            self._params['likelihood'].append(
                1 - self._params['confidence_level'])

        self._params['likelihood'].sort()
        self._upper_index = self._params['likelihood'].index(
            self._params['confidence_level'])
        self._lower_index = self._params['likelihood'].index(
            1 - self._params['confidence_level'])
        self._values_retriever.confidence_indexes = (
        self._lower_index, self._upper_index)

    def _clean_directory(self, exclusions : Optional[list] = None):
        clear_directory_content('lightning_logs')
        clear_directory_content(self._model_dir, exclusions)
        os.makedirs(self._model_dir, exist_ok=True)
        clear_directory_content(os.path.expanduser('~/ray_results'))

    def _obtain_data(self):

        if self._config["inputs"]["future_covariates"]['data']:
            self._futr_exog_list = \
                self._config["inputs"]["future_covariates"]['data']

            for dataset in self._datasets:
                input_future = read_csv_to_pd_formatted(
                    self._config["inputs"]["future_covariates"]["common"][
                        "model_data"][dataset], sort_by_column_name='time')
                for category in self._futr_exog_list :
                    input_future[category] = input_future[category].astype(str)
                setattr(self, f'_input_future_{dataset}', input_future)
        else:
            self._futr_exog_list = []

        for dataset in self._datasets:
            input_past = read_csv_to_pd_formatted(
                f"resources/input/model_data/input_past_{dataset}.csv")
            input_past.columns = input_past.columns.str.replace('.', '_',
                                                                regex=False)
            setattr(self, f'_input_past_{dataset}', input_past)

            output = read_csv_to_pd_formatted(f"resources/input/model_data/output_{dataset}.csv")
            setattr(self, f'_output_{dataset}', output)

    def _initialize_training_variables(self):
        if torch.cuda.is_available():
            self._accelerator = 'gpu'
            self._params['num_workers_loader'] = 4
        else :
            self._accelerator = 'auto'

    def _assign_data_to_models(self):
        self._future_predict = pd.DataFrame()
        for dataset_type in self._datasets:
            input_past = getattr(self, f'_input_past_{dataset_type}')
            input_future = getattr(self, f'_input_future_{dataset_type}',
                                   pd.DataFrame())
            output = getattr(self, f'_output_{dataset_type}')

            if not input_future.empty:
                if dataset_type.lower() == 'test':
                    self._future_predict = input_future
                input_future = input_future.drop(columns=['ds'])

            output = output.drop(columns=['ds'])
            if len(input_future) != len(input_past):
                raise ValueError("DataFrames do not have the same length.")

            data = pd.merge(pd.merge(input_past,input_future, on='time', how='outer'), output, on='time', how='outer')
            setattr(self, f'_{dataset_type}_data', data)
            setattr(self, f'_{dataset_type}_data', getattr(self, f'_{dataset_type}_data').assign(ds=pd.to_datetime(getattr(self, f'_{dataset_type}_data')['ds'])))

            missing_data = data.isnull().sum()
            empty_data = (data == '').sum()

            missing_locations = {col: data[data[col].isnull()].index.tolist() for col in data.columns if (missing_data[col] > 0).any()}
            empty_locations = {col: data[data[col] == ''].index.tolist() for col in data.columns if (empty_data[col] > 0).any()}

            if missing_locations:
                raise ValueError(f'Missing values in {dataset_type} data: {missing_locations}')
            if empty_locations:
                raise ValueError(f'Empty values in {dataset_type} data: {empty_locations}')


        self._hist_exog_list = [col for col in self._train_data.columns if col not in self._futr_exog_list and col
                                not in ["ds","time","y",] and not any(excluded in col for excluded in ["unique_id"])]


        unique_ids = self._train_data['unique_id'].unique()
        self._future_predict = pd.concat([
            pd.DataFrame({
                'unique_id': unique_id,
                **row._asdict()
            }, index=[0]) for row, unique_id in itertools.product(self._future_predict.itertuples(index=False), unique_ids)
        ])
        self._future_predict['ds'] = pd.to_datetime(self._future_predict['ds'])


class ModelBuilder(BaseModelBuilder):
    def __init__(
        self,
        config_manager: ConfigManager,
    ):
        super().__init__(config_manager)
        self._best_metrics = {}


    def run(self):

        for model in self._config["hyperparameters"]["models"]:
            self._initialize_variables()
            self._model_name = model
            self._model_dir =f'models/{self._model_name}'
            self._clean_directory()
            self._model_to_train =  CUSTOM_MODEL[self._model_name]
            self._obtain_data()
            self._assign_data_to_models()
            if self._config['common']['hyperparameters_optimization'][
                'is_optimizing']:
                self._assign_best_hyperparams()
            self._train_model()
            self._save_metrics_from_tensorboardflow()
            self._predict()
            self._delete_event_files()
            self._plot_predictions()
            self._coordinate_metrics_calculation()


    def _initialize_variables(self):
        self._params = self._assign_params()
        self._initialize_training_variables()
        self._lightning_logs_dir = 'lightning_logs'
        self._lower_index, self._upper_index = self._config_manager.confidence_indexes

    def _assign_best_hyperparams(self):
        optimized_model_path = self._model_dir.replace('models/','models/hyperparameters_optimization/')
        with open(f"{optimized_model_path}/best_study.pkl",
                'rb') as file:
            best_hyper_params = pickle.load(file)
        default_hypers = copy.deepcopy(self._config_manager.hyperparameters[self._model_name])
        default_params = copy.deepcopy(self._params)
        for hyper, value in default_params.items():
            if hyper in best_hyper_params.best_params:
                self._params[hyper] = best_hyper_params.best_params[hyper]

        self._adjust_likelihood()
        for hyper,value in default_hypers.items():
            if hyper in best_hyper_params.best_params:
                if hyper == 'loss':
                    self._config_manager.hyperparameters[self._model_name][hyper] = ConfigManager.assign_loss_fct(best_hyper_params.best_params,self._params)['loss']
                else :
                    self._config_manager.hyperparameters[self._model_name][hyper] = best_hyper_params.best_params[hyper]


    def _train_model(self, hyperparameter_phase: Optional[str] = 'hyperparameters'):
        hyperparam = self._config_manager.hyperparameters
        callbacks_list = []
        callbacks = self._config_manager.get_callbacks(self._model_name,hyperparameter_phase)['callbacks']
        for callback in callbacks:
            if isinstance(callback,
                          ModelCheckpoint):
                self._model_checkpoint = callback
                callbacks_list.append(self._model_checkpoint)
            if isinstance(callback,
                          EarlyStopping):
                callbacks_list.append(copy.deepcopy(callback))

        if not callbacks_list:
            callbacks_list = None

        if self._lightning_logs_dir:
            self._logger = TensorBoardLogger('')
        self._logger_dir = self._logger.log_dir
        keys_to_remove = {'likelihood', 'gradient_clip_val','confidence_level'}
        common_hypers = {key: value for key, value in self._params.items() if key not in keys_to_remove}
        nf = NeuralForecast(models = [
            self._model_to_train(
                valid_loss=RiskReturn(),
                futr_exog_list=self._futr_exog_list,
                hist_exog_list=self._hist_exog_list,
                enable_progress_bar=True,
                **hyperparam[self._model_name],
                **common_hypers,
                callbacks=callbacks_list,
                logger=self._logger,
                gradient_clip_val=self._params["gradient_clip_val"],
                enable_model_summary=True,
                enable_checkpointing=True,
                accelerator= self._accelerator,
                )
        ],
            freq=create_custom_trading_days(start_date=self._config['common']['start_date'],
                                            end_date = self._config['common']['end_date'])
)

        self._val_size = int(len(self._train_data)*self._config['common']['val_proportion_size']/hyperparam[self._model_name]['tgt_size'])
        nf.fit(df=self._train_data, val_size=self._val_size, use_init_models=True)
        nf.save(
            'lightning_logs/saved_nixtla',
            model_index=None,
            overwrite=True,
            save_dataset=True)
        self._copy_best_model()

    def _copy_best_model(self):
        nixtla_ckpt_name = None
        for filename in os.listdir('lightning_logs/saved_nixtla'):
            if filename.endswith('.ckpt'):
                nixtla_ckpt_name = filename
                break

        best_model_ckpt = None
        for filename in os.listdir(f'{self._logger_dir}/checkpoints'):
            if filename.endswith('.ckpt'):
                best_model_ckpt = filename
                break

        old_file_path = os.path.join(f'{self._logger_dir}/checkpoints', best_model_ckpt)
        new_file_path_in_checkpoints = os.path.join(f'{self._logger_dir}/checkpoints', nixtla_ckpt_name)
        os.rename(old_file_path, new_file_path_in_checkpoints)
        os.remove(f'lightning_logs/saved_nixtla/{nixtla_ckpt_name}')
        destination_path = os.path.join('lightning_logs/saved_nixtla', nixtla_ckpt_name)
        shutil.copyfile(new_file_path_in_checkpoints, destination_path)


    def _predict(self):
        best_nf_models = NeuralForecast.load('lightning_logs/saved_nixtla')
        test_size = len(self._test_data)
        self._y_hat_test = pd.DataFrame()
        current_train_data = self._train_data.copy()
        y_hat = best_nf_models.predict(current_train_data,futr_df=self._future_predict)
        self._y_hat_test = pd.concat([self._y_hat_test, y_hat.iloc[[-1]]])
        for i in range(test_size-1):
            combined_data = pd.concat([current_train_data, self._test_data.iloc[[i]]])
            y_hat = best_nf_models.predict(combined_data,futr_df=self._future_predict)
            self._y_hat_test = pd.concat([self._y_hat_test, y_hat.iloc[[-1]]])
            current_train_data = combined_data

        self._y_hat_test.reset_index(drop=True, inplace=True)
        if len(self._y_hat_test) != len(self._test_data):
            raise ValueError(f'Predicted vs target for test set are not the same length in {inspect.currentframe().f_code.co_name}')
        if not self._y_hat_test['ds'].equals(self._test_data['ds']):
            mismatched_rows = self._y_hat_test[self._y_hat_test['ds'] != self._test_data['ds']]
            raise ValueError(f'The dates in the predicted vs target sets do not match in {inspect.currentframe().f_code.co_name}. Mismatched rows:\n{mismatched_rows}')

        self._all_columns_except_ds = [col for col in self._y_hat_test.columns if col not in 'ds']
        self._median_column = [col for col in self._y_hat_test.columns if '-median' in col][0]
        self._quantile_cols = [col for col in self._y_hat_test.columns if col not in [self._median_column, 'ds']]


    def _delete_event_files(self):
        event_files = [f for f in os.listdir(self._logger_dir) if f.startswith('events.out.tfevents')]
        event_files.sort(key=lambda x: int(x.split('.')[3]))
        for file in event_files[1:]:
            os.remove(os.path.join(self._logger_dir, file))


    def _plot_predictions(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self._test_data['ds'], self._test_data['y'], color='black', label='Actual')
        plt.plot(self._y_hat_test['ds'], self._y_hat_test[self._median_column], label='Predicted', color='blue')
        plt.fill_between(self._y_hat_test['ds'], self._y_hat_test[self._quantile_cols[0]], self._y_hat_test[self._quantile_cols[-1]], color='gray', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Output')
        plt.title('Actual vs Predicted Values over time')
        plt.legend()
        plt.savefig(os.path.join(f'{self._model_dir}', 'actual_vs_predicted_test.png'))
        plt.close()

    def _coordinate_metrics_calculation(self):

        self._torch_target = torch.tensor(self._test_data['y'].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        self._torch_predicted = torch.tensor(self._y_hat_test[self._all_columns_except_ds].to_numpy(), dtype=torch.float32)
        daily_returns = MetricCalculation.gather_daily_returns(y=self._torch_target,
                                             y_hat=self._torch_predicted,
                                             lower_index=self._lower_index,
                                             upper_index=self._upper_index)

        self._metrics = MetricCalculation.get_risk_rewards_metrics(daily_returns,is_checking_nb_trades=False)
        self._metrics["nb_of_trades"] = daily_returns.shape[0]
        self._prepare_metrics_inputs()
        self._calculate_metrics()
        self._save_metrics()


    def _prepare_metrics_inputs(self):
        self._current_all_preds = []
        self._current_all_actuals = []
        self._preds_class = []
        self._actual_class = []
        self._cumulative_predicted_return = 1
        max_drawdown = 0
        peak = 1
        targets_size = len(self._torch_target)
        for item in range(targets_size):
            target, lower_return,upper_return = \
                MetricCalculation.convert_torch_to_list(y=self._torch_target,
                                                        y_hat=self._torch_predicted,
                                                        item=item,
                                                        lower_index=self._lower_index,
                                                        upper_index=self._upper_index)
            actual_return = target.item()

            median_pred_return = self._torch_predicted[item][len(self._torch_predicted[item])//2].item()


            if lower_return > 0 and upper_return > 0:
                self._cumulative_predicted_return *= (
                        1 + actual_return)
                self._preds_class.append(self._transform_return_to_class(median_pred_return))
                self._actual_class.append(self._transform_return_to_class(actual_return))

            elif upper_return < 0 and lower_return < 0:
                self._cumulative_predicted_return *= (1 - actual_return)
                self._preds_class.append(self._transform_return_to_class(median_pred_return))
                self._actual_class.append(self._transform_return_to_class(actual_return))

            if self._cumulative_predicted_return > peak:
                peak = self._cumulative_predicted_return
            else:
                drawdown = (peak - self._cumulative_predicted_return) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            self._current_all_preds.append(median_pred_return)
            self._current_all_actuals.append(actual_return)
        self._max_drawdown = max_drawdown


    @staticmethod
    def _transform_return_to_class(ret : float) -> int:
        return 1 if ret >=0 else 0


    def _calculate_metrics(self):
        self._f1_score = f1_score(self._actual_class, self._preds_class,
                                  average='weighted')
        self._rmse = np.sqrt(
            mean_squared_error(self._current_all_actuals, self._current_all_preds))
        if self._params["input_size"] <= 20:
            rolling_windows = self._params["input_size"] - 1
        else :
            rolling_windows = 20
        
        naive_forecast = []
        for i in range(len(self._torch_target)):
            if i < rolling_windows:
                naive_forecast.append(self._torch_target[i].item())
            else:
                window = self._torch_target[i-19:i+1]
                avg = torch.mean(window).item()
                naive_forecast.append(avg)

        self._naive_rmse = \
            np.sqrt(mean_squared_error(self._current_all_actuals, naive_forecast))
        actual_daily_returns = np.array(self._current_all_actuals)
        self._actual_annualized_return = (self._get_buy_and_hold()+1)** (252 / len(self._current_all_actuals)) - 1
        actual_annualized_risk = np.std(actual_daily_returns) * (252 ** 0.5)
        self._actual_return_on_risk = self._actual_annualized_return / actual_annualized_risk if actual_annualized_risk != 0 else 0


    def _get_buy_and_hold(self) -> float:
        asset = self._config['output'][0]['data'][0]['asset'].upper()
        output_for_asset = pd.read_csv(f'resources/input/preprocessed/{asset}_output.csv')
        evaluation_set = getattr(self,f'_test_data')
        first_date= (evaluation_set['ds'].iloc[0]).strftime("%Y-%m-%d")
        last_date = (evaluation_set['ds'].iloc[-1]).strftime("%Y-%m-%d")
        first_value = output_for_asset.loc[output_for_asset['ds'] == first_date, 'open'].iloc[0]
        last_value = output_for_asset.loc[output_for_asset['ds'] == last_date, 'close'].iloc[0]
        return last_value/first_value-1


    def _save_metrics(self):
        first_date = self._output_test['ds'].iloc[0]
        last_date = self._output_test['ds'].iloc[-1]


        self._aggregate_metrics = {
            "rmse": self._rmse,
            "f1_score": self._f1_score,
            "naive_forecast_rmse": self._naive_rmse,
            "rmse_vs_naive": self._rmse / self._naive_rmse if self._naive_rmse!=0 else 0,
            "annualized_return": self._metrics['annualized_return'].item(),
            "actual_annualized_return": self._actual_annualized_return,
            "ann_return_on_risk":self._metrics['return_on_risk'].item(),
            "ann_actual_return_on_risk": self._actual_return_on_risk,
            "max_drawdown": self._max_drawdown,
            "nb_of_trades":  self._metrics["nb_of_trades"],
            'first_last_ds' : (first_date, last_date)
        }


        metrics_path = os.path.join(
            f'{self._model_dir}', 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self._aggregate_metrics, f, ensure_ascii=False, indent=4)

    def _save_metrics_from_tensorboardflow(self):
        metrics_dict = {}

        os.makedirs(f'{self._model_dir}/tensorboard', exist_ok=True)
        for event_file in os.listdir(self._logger.log_dir):
            if not event_file.startswith('events.out.tfevents'):
                continue
            full_path = os.path.join(self._logger.log_dir, event_file)
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

            values = [scalar.value for scalar in scalars]
            plt.plot(steps, values, label=metric)
            plt.xlabel('Steps' if metric == 'train_loss_step' else 'Epoch')
            plt.ylabel('Value')
            plt.title(metric)
            plt.legend(loc='upper right')
            plt.savefig(f"{self._model_dir}/tensorboard/{metric.replace('/', '_')}.png")
            plt.close()
