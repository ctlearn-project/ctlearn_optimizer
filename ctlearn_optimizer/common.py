import os
import re
import shutil
import csv
import pickle
import logging
from timeit import default_timer as timer
from time import sleep
from multiprocessing import Pool
import numpy as np
import sklearn.metrics
import yaml
from ctlearn.run_model import run_model


def set_value(dictionary, value, *keys):
    """Modify the value the keys point to in a nested dictionary

    dictionary can be a nested dictionary containing lists, this lists can also
    contain nested dictionaries, and so on. *keys list can contain strings
    (which refer dictionary keys) and integers (which refer list indices).
    dictionary cannot be empty

    Args:
        dictionary: dictionary that contais the key-value pair [dict]
        value: value to set [int, float, string]
        *keys: list of keys containing strings and integers [list]

    Returns:
        modified dictionary [dict]

    Example:
        dictionary = {'a':[0,{'b':1},0]}
        value = 2
        *keys = ['a', 1, 'b']
        set_value(dictionary, value, *keys) = {'a':[0,{'b':2},0]}
    """
    if type(dictionary) is not dict:
        raise TypeError('set_value expects dict as first argument.')

    _keys = keys[:-1]
    _element = dictionary
    for key in _keys:
        _element = _element[key]
    _element[keys[-1]] = value

    return dictionary


def create_nested_item(dictionary, *keys):
    """Create an empty item with specific keys and positions in a dictionary

    dictionary can be a nested dictionary containing lists, this lists can also
    contain nested dictionaries, and so on. *keys list can contain strings
    (which refer dictionary keys) and integers (which refer list indices).
    dictionary may or may be not empty.

    Args:
        dictionary: dictionary to modify [dict]
        *keys: list of keys containing strings and integers [list]

    Returns:
        modified dictionary [dict]

    Example:
        dictionary = {}
        *keys = ['a', 'b', 1, 'c', 2 , 'd']
        create_nested_item(dictionary, *keys) =
        {'a': {'b': [0, {'c': [0, 0, {'d': {}}]}]}}
    """

    if type(dictionary) is not dict:
        raise TypeError('create_nested_item() expects dict as first argument.')

    _keys = keys
    _element = dictionary

    # iterate over the list of keys
    for n in range(len(_keys)):
        key = _keys[n]
        # set next_key value
        if n < len(_keys) - 1:
            next_key = _keys[n+1]
        else:
            next_key = None

        if type(key) is str:
            if type(_element) is dict:
                # if key in the dict, access the item
                if key in _element:
                    _element = _element[key]
                # else, create the item
                else:
                    if type(next_key) is str:
                        _element.update({'{}'.format(key): {}})
                        _element = _element[key]
                    if type(next_key) is int:
                        _element.update({'{}'.format(key): []})
                        _element = _element[key]
                    if next_key is None:
                        _element.update({'{}'.format(key): {}})

        if type(key) is int:
            if type(_element) is list:
                # try to access list element
                try:
                    _dummy_element = _element[key]
                    if type(next_key) is str and type(_dummy_element) is not dict:
                        _element[key] = {}
                    if type(next_key) is int and type(_dummy_element) is not list:
                        _element[key] = []
                    _element = _element[key]
                # except, create list element
                except:
                    # if list length is enough
                    if len(_element) >= key + 1:
                        if type(next_key) is str:
                            _element[key] = {}
                        if type(next_key) is int:
                            _element[key] = []
                    # else, append 0 elements to the list
                    else:
                        while len(_element) < key + 1:
                            _element.append(0)
                        if type(next_key) is str:
                            _element[key] = {}
                        if type(next_key) is int:
                            _element[key] = []

                    _element = _element[key]

    return dictionary


def auxiliar_modify_params(self, hyperparams):
    """Set hyperparameters values in ctlearn config file

    Args:
        self
        hyperparams: dictionary containing key-value pairs of hyperparameters [dict]
    """
    # load ctlearn config file
    with open(self.ctlearn_config, 'r') as config:
        myconfig = yaml.load(config)

    # empty layers list in myconfig in order to get rid of previous
    # configurations
    myconfig['Model']['Model Parameters']['basic']['conv_block']['layers'] = []

    hyperparameters_config = self.opt_config['Hyperparameters']['Config']
    # modify hyperparameters values in myconfig
    for param, value in hyperparams.items():
        if param in hyperparameters_config:
            # create hyperparameter empty item in myconfig
            create_nested_item(myconfig, *hyperparameters_config[param])
            # set hyperparameter value
            set_value(myconfig, value, *hyperparameters_config[param])

    iteration = self.iteration
    # set model and prediction_file_path
    myconfig['Logging']['model_directory'] = './run' + str(iteration)
    myconfig['Prediction']['prediction_file_path'] = \
        './run' + str(iteration) + '/predictions_run{}.csv'.format(iteration)

    # dump ctlearn configuration
    with open(self.ctlearn_config, 'w') as config:
        yaml.dump(myconfig, config)


def get_pred_metrics(self):
    """Get prediction set metrics

    Args:
        self

    Returns:
        dictionary containing prediction set metrics [dict]

    """

    iter = self.iteration
    user_defined_metric_pred = self.opt_config.get('user_defined_metric_pred',
                                                   None)
    list_metrics_pred_to_log = self.opt_config.get('metrics_pred_to_log', [])

    # load prediction.csv
    predictions_path = './run' + str(iter) + \
        '/predictions_run{}.csv'.format(iter)
    predictions = np.genfromtxt(predictions_path, delimiter=',', names=True)
    labels = predictions['gamma_hadron_label'].astype(int)
    gamma_classifier_values = predictions['gamma']
    predicted_class = predictions['predicted_class'].astype(int)

    # compute metrics
    fpr, tpr, _thresholds = sklearn.metrics.roc_curve(
        labels, gamma_classifier_values, pos_label=0)
    auc = sklearn.metrics.auc(fpr, tpr)
    f1 = sklearn.metrics.f1_score(labels, predicted_class)
    acc = sklearn.metrics.accuracy_score(labels, predicted_class)
    bacc = sklearn.metrics.balanced_accuracy_score(labels, predicted_class)
    prec = sklearn.metrics.precision_score(labels, predicted_class)
    rec = sklearn.metrics.recall_score(labels, predicted_class)
    log_loss = sklearn.metrics.log_loss(labels, predicted_class)

    metrics_pred = {'auc': auc, 'acc': acc, 'bacc': bacc, 'f1': f1,
                    'prec': prec, 'rec': rec, 'log_loss': log_loss}

    # compute validation user defined metric if it's not None
    if user_defined_metric_pred is not None:
        if user_defined_metric_pred['expression'] is not None:
            user_defined = eval(user_defined_metric_pred['expression'])
            metrics_pred.update(
                {user_defined_metric_pred['label']: user_defined})

    # return metrics_pred_to_log
    metrics_pred_to_log = {}
    for metric in list_metrics_pred_to_log:
        metrics_pred_to_log.update(
            {metric + '_pred': metrics_pred.get(metric)})

    return metrics_pred_to_log


def get_val_metrics(self):
    """Get validation set metrics

    Args:
        self

    Returns:
        dictionary containing validation set metrics [dict]
    """

    iter = self.iteration
    user_defined_metric_val = self.opt_config.get(
        'user_defined_metric_val', None)
    list_metrics_val_to_log = self.opt_config.get('metrics_val_to_log', [])

    # load training log file
    run_folder = './run' + str(iter)
    for file in os.listdir(run_folder):
        if file.endswith('logfile.log'):
            with open(run_folder + '/' + file) as log_file:
                contents = log_file.read()
                # ensure that prediction log file is not loaded
                if 'Training' in contents:
                    train_logfile = file

    with open(run_folder + '/' + train_logfile, 'r') as stream:
        r = re.compile('INFO:Saving dict for global step .*')
        matches = list(filter(r.match, stream))
        assert len(matches) > 0
        val_info = matches[-1]
    # extract validation metrics
    auc = float(re.findall(r'auc = [-+]?\d*\.*\d+', val_info)[0][6:])
    acc = float(re.findall(r'accuracy = [-+]?\d*\.*\d+', val_info)[0][11:])
    acc_gamma = float(re.findall(
        r'accuracy_gamma = [-+]?\d*\.*\d+', val_info)[0][17:])
    acc_proton = float(re.findall(
        r'accuracy_proton = [-+]?\d*\.*\d+', val_info)[0][18:])
    loss = float(re.findall(r'loss = [-+]?\d*\.*\d+', val_info)[0][7:])

    metrics_val = {'auc': auc, 'acc': acc, 'acc_gamma': acc_gamma,
                   'acc_proton': acc_proton, 'loss': loss}

    # compute prediction user defined metric if it's not None
    if user_defined_metric_val is not None:
        if user_defined_metric_val['expression'] is not None:
            user_defined = eval(
                user_defined_metric_val['expression'], metrics_val)
            metrics_val.update(
                {user_defined_metric_val['label']: user_defined})

    # return metrics_val_to_log
    metrics_val_to_log = {}
    for metric in list_metrics_val_to_log:
        metrics_val_to_log.update({metric + '_val': metrics_val.get(metric)})

    return metrics_val_to_log


def set_initial_config(self):
    """Set basic config and fixed hyperparameters in ctlearn config file

    Args:
        self
    """

    # load ctlearn config file
    with open(self.ctlearn_config, 'r') as config:
        myconfig = yaml.load(config)

    # set basic configuration
    basic_config = self.opt_config['Basic_config']
    myconfig['Training']['num_validations'] = basic_config['num_validations']
    myconfig['Training']['num_training_steps_per_validation'] = \
        basic_config['num_training_steps_per_validation']
    myconfig['Data']['Input']['batch_size'] = basic_config['batch_size']
    myconfig['Model']['model_directory'] = basic_config['model_directory']
    myconfig['Data']['Loading']['validation_split'] = \
        basic_config['validation_split']
    myconfig['Data']['Processing']['sorting'] = basic_config['sorting']
    myconfig['Data']['Loading']['min_num_tels'] = basic_config['min_num_tels']
    myconfig['Data']['Loading']['example_type'] = basic_config['example_type']

    if basic_config['model'] == 'cnn_rnn':
        myconfig['Model']['model']['module'] = 'cnn_rnn'
        myconfig['Model']['model']['function'] = 'cnn_rnn_model'
        assert basic_config['example_type'] == 'array'

    elif basic_config['example_type'] == 'single_tel':
        myconfig['Model']['model']['module'] = 'single_tel'
        myconfig['Model']['model']['function'] = 'single_tel_model'
        assert basic_config['example_type'] == 'single_tel'

    myconfig['Data']['Loading']['selected_tel_types'] = \
        basic_config['selected_tel_types']
    aux_dict = {'SST:ASTRICam': {'camera_types': 'ASTRICam',
                                 'interpolation_image_shape': [56, 56, 1]},
                'SST:CHEC': {'camera_types': 'CHEC',
                             'interpolation_image_shape': [48, 48, 1]},
                'SST:DigiCam': {'camera_types': 'DigiCam',
                                'interpolation_image_shape': [96, 96, 1]},
                'MST:FlashCam': {'camera_types': 'FlashCam',
                                 'interpolation_image_shape': [112, 112, 1]},
                'LST:LSTCam': {'camera_types': 'LSTCam',
                               'interpolation_image_shape': [110, 110, 1]},
                'MST:NectarCam': {'camera_types': 'NectarCam',
                                  'interpolation_image_shape': [110, 110, 1]},
                'SCT:SCTCam': {'camera_types': 'SCTCam',
                               'interpolation_image_shape': [120, 120, 1]}}

    myconfig['Image Mapping']['camera_types'] = []
    myconfig['Image Mapping']['interpolation_image_shape'] = {}
    for tel_type in basic_config['selected_tel_types']:
        element = aux_dict[tel_type]
        myconfig['Image Mapping']['camera_types'].append(
            element['camera_types'])
        myconfig['Image Mapping']['interpolation_image_shape'].update(
            {element['camera_types']: element['interpolation_image_shape']})

    # set fixed hyperparameters values
    hyperparameter_dict = self.opt_config['Hyperparameters']
    fixed_hyperparameters = hyperparameter_dict.get(
        'Fixed_hyperparameters', None)
    hyperparameters_config = hyperparameter_dict.get('Config')

    if hyperparameters_config is None:
        raise KeyError('hyperparameters_config is empty')
    if fixed_hyperparameters is not None:
        for param, value in fixed_hyperparameters.items():
            create_nested_item(myconfig, *hyperparameters_config[param])
            set_value(myconfig, value, *hyperparameters_config[param])

    # dump ctlearn configuration
    with open(self.ctlearn_config, 'w') as config:
        yaml.dump(myconfig, config)


def run_train(config):
    """Auxiliar function for train()

    debug is set to False and log_to_file is set to True

    Args:
        config: loaded ctlearn config file
    """

    run_model(config, mode='train', debug=False, log_to_file=True)


def train(self):
    """Run a CTlearn model training

    Args:
        self
    """

    # modify file_list in myconfig
    with open(self.ctlearn_config, 'r') as config:
        myconfig = yaml.load(config)

    basic_config = self.opt_config['Basic_config']
    myconfig['Data']['file_list'] = basic_config['training_file_list']

    # dump ctlearn configuration
    with open(self.ctlearn_config, 'w') as config:
        yaml.dump(myconfig, config)

    # run training as subprocess to free GPU memory after each run
    with Pool(1) as p:
        p.apply(run_train, (myconfig,))


def run_pred(config):
    """Auxiliar function for predict()

    debug is set to False and log_to_file is set to True

    Args:
        config: loaded ctlearn config file
    """

    run_model(config, mode='predict', debug=False, log_to_file=True)


def predict(self):
    """Predict using a trained CTLearn model

    Args:
        self
    """

    # modify file_list in myconfig
    with open(self.ctlearn_config, 'r') as config:
        myconfig = yaml.load(config)

    basic_config = self.opt_config['Basic_config']
    myconfig['Data']['file_list'] = basic_config['prediction_file_list']

    # modify ctlearn config to make sure that prediction file will be created
    myconfig['Prediction']['export_as_file'] = True
    myconfig['Prediction']['true_labels_given'] = True

    # dump ctlearn configuration
    with open(self.ctlearn_config, 'w') as config:
        yaml.dump(myconfig, config)

    # run prediction as subprocess to free GPU memory after each run
    with Pool(1) as p:
        p.apply(run_pred, (myconfig,))


def modify_optimizable_params(self, hyperparams):
    """Update ctlearn config file with new hyperparameters each iteration

    Args:
        self
        hyperparams: flat or nested dictionary containing hyperparameters to
                     optimize generated by hyperopt or skopt [dict]

    Returns:
        flat dictionary containing  dependent hyperparameters and
        hyperparameters to optimize [dict]
    """

    # flatten optimizable hyperparameters dict if required
    def aux_flat(hyperparams):
        flat_hyperparams = {}
        for key, item in hyperparams.items():
            if type(item) is not dict:
                flat_hyperparams.update({key: item})
            else:
                flat_hyperparams.update(aux_flat(item))
        return flat_hyperparams

    hyperparams = aux_flat(hyperparams)

    # correct hyperparameters_to_optimize keys (hyperopt space creator doesn't
    # support repeated labels, so a ! character is appended to each repeated
    # label)
    corrected_hyperparams = {}
    for key in hyperparams:
        if key.endswith('!'):
            dummy_key = key
            while dummy_key.endswith('!'):
                dummy_key = dummy_key[:-1]
            corrected_hyperparams.update({dummy_key: hyperparams[key]})
        else:
            corrected_hyperparams.update({key: hyperparams[key]})

    hyperparams = corrected_hyperparams

    # add dependent hyperparameters to the hyperparameters dict
    hyperparameters_config = self.opt_config['Hyperparameters']
    dependent_hyperparameters = hyperparameters_config.get(
        'Dependent_hyperparameters', None)
    if dependent_hyperparameters is not None:
        for param, expression in dependent_hyperparameters.items():
            hyperparams.update({param: eval(expression, hyperparams)})

    # update myconfig with the values in hyperparameters dict
    auxiliar_modify_params(self, hyperparams)

    return hyperparams


def objective(self, hyperparams):
    """Objective function to minimize

    Args:
        self
        hyperparams: set of hyperparameters to evaluate  [dict]

    Returns:
        metric to minimize [float]
    """

    self.iteration += 1
    self.counter += 1

    logging.info('Iteration: {0}' .format(self.counter))
    logging.info('Global iteration: {0}' .format(self.iteration))

    # update hyperparameters
    hyperparams_dict = modify_optimizable_params(self, hyperparams)

    start = timer()
    logging.info('Training')

    # train ctlearn network
    # this tries to solve a problem with the loading of the training data
    dummy = 1
    for _ in range(0, 9):
        # try to train the model
        try:
            train(self)
            dummy = None
        # log the error and remove the model directory created
        except Exception as str_error:
            logging.info('An error has occurred: {0}'.format(str_error))
            run_folder = './run' + str(self.iteration)
            shutil.rmtree(run_folder, ignore_errors=True)
        if dummy:
            sleep(10)
        else:
            break

    logging.info('Training ended')
    run_time = timer() - start

    # get validation set metrics
    metrics_val = get_val_metrics(self)
    metrics_pred = {}

    # predict if required
    if self.opt_config['predict']:
        logging.info('Predicting')
        predict(self)
        logging.info('Prediction ended')
        metrics_pred = get_pred_metrics(self)

    # set loss depending on metric and data set to optimize
    if self.opt_config['data_set_to_optimize'] == 'validation':
        metric = self.opt_config['metric_to_optimize'] + '_val'
        loss = 1 - metrics_val[metric]
        logging.info('{0}: {1}'.format(metric, metrics_val[metric]))

    elif self.opt_config['data_set_to_optimize'] == 'prediction':
        metric = self.opt_config['metric_to_optimize'] + '_pred'
        loss = 1 - metrics_pred[metric]
        logging.info('{0}: {1}'.format(metric, metrics_pred[metric]))

    # write hyperparameters and metrics in the checking_file
    hyper_to_log = self.opt_config['Hyperparameters']['Hyperparameters_to_log']
    with open('./checking_file.csv', 'a') as file:
        writer = csv.writer(file)
        row_hyperparams = []

        for element in hyper_to_log:
            if element in hyperparams_dict:
                row_hyperparams.append(hyperparams_dict[element])
            else:
                row_hyperparams.append(0)

        row = [loss, self.iteration] + row_hyperparams + \
            list(metrics_val.values()) + \
            list(metrics_pred.values()) + [run_time]
        writer.writerow(row)

    # save checking_trials file for resuming training if it has been interrupted
    pickle.dump(self.trials, open("checking_trials.pkl", "wb"))

    # remove training folder in order to avoid space issues in long runs
    if not self.keep_training_folders:
        run_folder = './run' + str(self.iteration)
        shutil.rmtree(run_folder, ignore_errors=True)

    return loss
