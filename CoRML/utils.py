r"""
    Name: utils.py
    Date: 2023/04/12
    Description: Utility functions.
"""

import os
import shutil
import warnings
from logging import getLogger
from datetime import datetime
from recbole.config import Config
from recbole.trainer import Trainer
from recbole.data.dataset import Dataset
from recbole.data import data_preparation
from recbole.utils import init_seed
from recbole.utils.enum_type import ModelType

from CoRML.model import CoRML

warnings.filterwarnings('ignore')

class RecTrainer(object):
    r"""
    Trainer class.
    """
    def __init__(self, dataset='yelp2018'):
        self.model_name = CoRML
        self.dataset = dataset
        self.dataset_file_list = ['Params/Overall.yaml', 'Params/{:s}.yaml'.format(dataset)]
        self.preprocessing()

    def preprocessing(self):
        self.config = Config(model=self.model_name, config_dict=None, config_file_list=self.dataset_file_list)
        init_seed(self.config['seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        dataset = Dataset(self.config)
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, dataset)

    def train(self, verbose=False):
        self.logger.info("Start training:")
        model = self.model_name(self.config, self.train_data.dataset).to(self.config['device'])
        self.trainer = Trainer(self.config, model)
        shutil.rmtree(self.trainer.tensorboard.log_dir)
        cur_time = datetime.now().strftime('%b-%d-%Y_%H-%M-%S')
        self.trainer.tensorboard = get_tensorboard(cur_time, self.config['model'])
        _, _ = self.trainer.fit(self.train_data, None, verbose=verbose, show_progress=False)
        load_best = False if self.trainer.model.type == ModelType.TRADITIONAL else True
        self.logger.info("Start evalutaion:")
        test_result = self.trainer.evaluate(self.test_data, load_best_model=load_best)
        os.remove(self.trainer.saved_model_file)
        shutil.rmtree(os.path.join('Log/{:s}'.format(self.config['model']), cur_time))
        if os.path.exists('log_tensorboard'):
            shutil.rmtree(os.path.join('log_tensorboard'))
        for k, v in test_result.items():
            self.logger.info("{:12s}: {:.6f}".format(k, v))

from torch.utils.tensorboard import SummaryWriter
from recbole.utils import ensure_dir

def get_tensorboard(cur_time, model):
    r"""
    Modified version of get_tensorboard in Recbole.
    Source: https://github.com/RUCAIBox/RecBole/blob/5d7df69bcbe9d21b4185946e8ee9a4bd8f041b9d/recbole/utils/utils.py#L206-L230
    """
    base_path = 'Log/{:s}'.format(model)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    dir_name = cur_time
    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer

import logging
import colorlog
from colorama import init
from logging import getLogger
from recbole.utils.logger import log_colors_config, RemoveColorFilter

def init_logger(config):
    r"""
    Modified version of init_logger in Recbole.
    Source: https://github.com/RUCAIBox/RecBole/blob/5d7df69bcbe9d21b4185946e8ee9a4bd8f041b9d/recbole/utils/logger.py#L60-L118
    """
    init(autoreset=True)
    LOGROOT = './Log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)

    logfilename = 'RUN.log'

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])