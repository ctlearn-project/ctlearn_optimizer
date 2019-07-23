"""Automated model optimization framework for CTLearn"""
import pickle
import logging
import argparse
import os
import csv
from functools import partial
import numpy as np
import yaml
import hyperopt
import skopt
import ctlearn_optimizer.bayesian_tpe as bayesian_tpe
import ctlearn_optimizer.bayesian_gp as bayesian_gp
import ctlearn_optimizer.common as common


class optimizer:
    """ Basic class for an optimizer

    Methods:
        create_space_params: return hyperparameters space with required style
        set_initial_config: set basic config and fixed hyperparameters
        get_val_metrics: return validation set metrics
        get_pred_metrics: return prediction set metrics
        train: train a CTlearn model
        predict: predict using a trained CTLearn model
        objective: return objective function to optimize with required style
        optimize: start the optimization
    """

    def __init__(self, opt_config):
        """ Initialize the class

        Load trials file or create one as required. A trials file allows to
        resume an optimization run.
        Load checking_file or create one as required. The checking file is
        where the results of the optimization run (loss, iteration, metrics,
        hyperparameters, time) are stored.

        Args:
            opt_config: loaded optimization configuration file
        """

        self.counter = 0
        self.opt_config = opt_config
        self.random_state = opt_config.get('random_state', None)
        self.ctlearn_config = opt_config['ctlearn_config']
        self.n_startup_jobs = opt_config.get('n_startup_jobs', 20)
        self.optimization_type = opt_config['optimization_type']
        self.num_max_evals = opt_config['num_max_evals']
        self.reload_trials = opt_config['reload_trials']
        self.reload_checking_file = opt_config['reload_checking_file']
        self.gaussian_processes_config = opt_config.get(
            'gaussian_processes_config', {})
        self.keep_training_folders = opt_config.get(
            'keep_training_folders', False)

        if self.opt_config['data_set_to_optimize'] == 'prediction':
            assert self.opt_config['predict'] is True

        if self.optimization_type in ['tree_parzen_estimators', 'random_search']:
            # load trials file if reload_trials is True
            if self.reload_trials:
                assert os.path.isfile('trials.pkl')
                self.trials = pickle.load(open('trials.pkl', 'rb'))
                logging.info('Found trials.pkl file with {} saved trials'.format(
                    len(self.trials.trials)))
                # set iteration and num_max_evals to match load trials
                self.num_max_evals += len(self.trials.trials)
                self.iteration = len(self.trials.trials)
            # else, create trials file
            else:
                self.trials = hyperopt.Trials()
                logging.info('No trials file loaded, starting from scratch')
                self.iteration = 0

        elif self.optimization_type in ['gaussian_processes']:
            # load trials file if reload_trials is True
            if self.reload_trials:
                assert os.path.isfile('trials.pkl')
                self.trials = pickle.load(open('trials.pkl', 'rb'))
                logging.info('Found trials.pkl file with {} saved trials'.format(
                    len(self.trials.Xi)))
                # set iteration to match load trials
                self.iteration = len(self.trials.Xi)
            # else, create trials file
            else:
                hyperparameter_space = self.create_space_hyperparams()
                gp_c = self.gaussian_processes_config
                self.trials = skopt.Optimizer(hyperparameter_space,
                                              n_initial_points=self.n_startup_jobs,
                                              base_estimator=gp_c.get(
                                                  'base_estimator', 'GP'),
                                              acq_func=gp_c.get(
                                                  'acq_function', 'gp_hedge'),
                                              acq_optimizer=gp_c.get(
                                                  'acq_optimizer', 'auto'),
                                              random_state=self.random_state,
                                              acq_func_kwargs={'xi': gp_c.get('xi', 0.01),
                                                               'kappa': gp_c.get('kappa', 1.96)})
                logging.info('No trials file loaded, starting from scratch')
                self.iteration = 0

        # load checking_file.csv if reload_checking_file is True
        if self.reload_checking_file:
            assert os.path.isfile('./checking_file.csv')
            with open('./checking_file.csv', 'r') as file:
                existing_iters_csv = len(file.readlines()) - 1

            logging.info('Found checking_file.csv with {} saved trials, \
                         new trials will be added'.format(existing_iters_csv))

            if existing_iters_csv != self.iteration:
                logging.info(
                    'Caution: the number of saved trials in trials.pkl and \
                     checking_file.csv files  does not match')
        # else, create checking_file.csv
        else:
            logging.info(
                'No checking_file.csv file loaded, starting from scratch')

            hyperparams_to_log = \
                self.opt_config['Hyperparameters']['Hyperparameters_to_log']
            list_metrics_val_to_log = self.opt_config.get(
                'metrics_val_to_log', [])
            list_metrics_pred_to_log = self.opt_config.get(
                'metrics_pred_to_log', [])
            with open('./checking_file.csv', 'w') as file:
                writer = csv.writer(file)
                header = ['loss', 'iteration'] + hyperparams_to_log + \
                    [elem + '_val' for elem in list_metrics_val_to_log] + \
                    [elem + '_pred' for elem in list_metrics_pred_to_log] + \
                    ['run_time']

                writer.writerow(header)

    def create_space_hyperparams(self):
        """ Return hyperparameters space following required style

        Currently, only tree_parzen_estimators and random_search
        using hyperopt and gaussian_processes using skopt are supported

        Returns:
            hyperparameters space following hyperopt or skopt syntax

        Raises:
            NotImplementedError if self.optimization_type is other than
            tree_parzen_estimators, random_search or gaussian_processes
        """

        if self.optimization_type == 'tree_parzen_estimators':
            hyperparameter_space = bayesian_tpe.hyperopt_space(self)
        elif self.optimization_type == 'random_search':
            hyperparameter_space = bayesian_tpe.hyperopt_space(self)
        elif self.optimization_type == 'gaussian_processes':
            hyperparameter_space = bayesian_gp.skopt_space(self)
        else:
            raise NotImplementedError(
                'Other optimization types are not supported yet')
        return hyperparameter_space

    def set_initial_config(self):
        """Set basic config and fixed hyperparameters in ctlearn config file
        """
        common.set_initial_config(self)

    def get_pred_metrics(self):
        """Get prediction set metrics

        Returns:
            dictionary containing prediction set metrics
        """
        return common.get_pred_metrics(self)

    def get_val_metrics(self):
        """Get validation set metrics

        Returns:
            dictionary containing validation set metrics
        """
        return common.get_val_metrics(self)

    def train(self):
        """Train a CTLearn model
        """
        common.train(self)

    def predict(self):
        """Predict using a trained CTLearn model
        """
        common.predict(self)

    def objective(self, hyperparams):
        """ Return objective function to optimize following required style

        Currently, only tree_parzen_estimators and random_search
        using hyperopt and gaussian_processes using skopt are supported

        Returns:
            objective function for hyperopt or skopt input - output workflow

        Raises:
            NotImplementedError if self.optimization_type is other than
            tree_parzen_estimators, random_search or gaussian_processes
        """
        if self.optimization_type == 'tree_parzen_estimators':
            objective = bayesian_tpe.objective(self, hyperparams)
        elif self.optimization_type == 'random_search':
            objective = bayesian_tpe.objective(self, hyperparams)
        elif self.optimization_type == 'gaussian_processes':
            # set decorator in order to keep the labels of the hyperparameters
            @skopt.utils.use_named_args(self.create_space_hyperparams())
            def aux_gp_objective(**hyperparams):
                return bayesian_gp.objective(self, hyperparams)
            objective = aux_gp_objective(hyperparams)
        else:
            raise NotImplementedError(
                'Other optimization types are not supported yet')
        return objective

    def optimize(self):
        """ Start the optimization

        Currently, only tree_parzen_estimators and random_search
        using hyperopt and gaussian_processes using skopt are supported

        Raises:
            NotImplementedError if self.optimization_type is other than
            tree_parzen_estimators, random_search or gaussian_processes
        """

        # set initial config and get hyperparameter_space
        self.set_initial_config()

        # select otimization algorithm for hyperopt
        if self.optimization_type == 'tree_parzen_estimators':
            algo = partial(hyperopt.tpe.suggest,
                           n_startup_jobs=self.n_startup_jobs)
        if self.optimization_type == 'random_search':
            algo = hyperopt.rand.suggest

        # call optimizator
        if self.optimization_type in ['tree_parzen_estimators', 'random_search']:
            # set random state
            my_rstate = np.random.RandomState(self.random_state)

            hyperparameter_space = self.create_space_hyperparams()
            _fmin = hyperopt.fmin(self.objective, hyperparameter_space,
                                  algo, trials=self.trials,
                                  max_evals=self.num_max_evals, rstate=my_rstate)

        elif self.optimization_type in ['gaussian_processes']:
            gp_model = self.trials.run(self.objective, self.num_max_evals)
            pickle.dump(gp_model, open('gp_model.pkl', 'wb'))

        # save trials file
        pickle.dump(self.trials, open('trials.pkl', 'wb'))
        logging.info('trials.pkl saved')
        logging.info('Optimization run finished')


###################
# launch optimization
###################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=('Run Ctlearn model optimization'))
    parser.add_argument(
        'opt_config',
        help='path to YAML file containing ctlearn_optimizer configuration')
    args = parser.parse_args()

    log_file = 'optimization.log'
    logging.basicConfig(level=logging.INFO, filename=log_file)
    consoleHandler = logging.StreamHandler(os.sys.stdout)
    logging.getLogger().addHandler(consoleHandler)
    logging.info('Starting optimization run')

    with open(args.opt_config, 'r') as opt_config:
        opt_config = yaml.load(opt_config)

    model = optimizer(opt_config)
    model.optimize()
