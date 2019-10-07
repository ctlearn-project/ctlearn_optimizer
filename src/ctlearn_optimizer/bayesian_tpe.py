import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope


def hyperopt_space(hyper_to_opt):
    """Create space of hyperparameters for the tree parzen estimators optimizer.

    This function creates the space of hyperparameter following hyperopt
    syntax.

    Parameters:
        hyper_to_opt (dict): dictionary containing the configuration of the
            hyperparameters to optimize. This dictionary must follow the next
            syntax:

            .. code:: python

                hyper_to_opt = {'hyperparam_1': {'type': ...,
                                                 'range: ...,
                                                 'step': ...},
                                'hyperparam_2': {'type': ...,
                                                 'range: ...,
                                                 'step': ...},
                                ...
                                }

            See the oficial documentation for more details.

    Returns:
        dict: space of hyperparameters following the syntax required by
        the tree parzen estimators optimization algorithm.

    Example::

        hyper_top_opt = {
            'cnn_rnn_dropout':{
                'type': 'uniform',
                'range': [0,1]},
            'optimizer_type': {
                'type': 'choice',
                'range': ['Adadelta', 'Adam', 'RMSProp', 'SGD']},
            'number_of_layers':{
                'type': 'conditional',
                'range': {
                    'value': 1,
                    'cond_params':{
                        'layer1_kernel':{
                            'type': 'quniform',
                            'range': [2, 10],
                            'step': 1},
                        'base_learning_rate':{
                            'type': 'loguniform',
                            'range': [-5, 0]} }}}}

    """

    def aux_hyperopt(key, typee, rangee, keys_list, step):
        """Return a single hyperparameter element.

        This auxiliar function returns a dictionary that defines a space of
        hyperparameters containing a single hyperparameter following hyperopt
        syntax. A whole space of hyperparameters can be build by updating it
        with the output of this function.

        Parameters:
            key (string): hyperparameter label.
            typee (string): hyperparameter type.
            rangee (list): hyperparameter range.
            keys_list (list): list containing the hyperparameter labels that
                have already been added to the space.
            step (int): hyperparameter step, used for q-types.

        Returns:
            element (dict): space of hyperparameters containing a single
                hyperparameter element.
            keys_list (list): updated list containing the hyperparameter
                labels.
    """
        # create auxiliary dict
        dict_type = {'uniform': hp.uniform,
                     'quniform': hp.quniform,
                     'loguniform': hp.loguniform,
                     'qloguniform': hp.qloguniform,
                     'normal': hp.normal,
                     'qnormal': hp.qnormal,
                     'lognormal': hp.lognormal,
                     'qlognormal': hp.qlognormal,
                     'choice': hp.choice,
                     'conditional': hp.choice}

        # hyperparameters of different types must be handled in different ways
        not_log_type_strings = ('uniform', 'quniform', 'normal', 'qnormal')
        log_type_strings = ('loguniform', 'qloguniform',
                            'lognormal', 'qlognormal')
        step_type_strings = ('qloguniform', 'qloguniform',
                             'quniform', 'qnormal')

        if typee in not_log_type_strings:
            if typee not in step_type_strings:
                element = {key: dict_type[typee](key,
                                                 rangee[0],
                                                 rangee[1])}
            # hyperparameters belonging to step_type_strings have an additional
            # step parameter and their results need to be converted back into
            # int type
            else:
                element = {key: scope.int(dict_type[typee](key,
                                                           rangee[0],
                                                           rangee[1],
                                                           step))}
        # the range of hyperparameters belonging to log_type_strings must be
        # modified
        elif typee in log_type_strings:
            if typee not in step_type_strings:
                element = {key: dict_type[typee](key,
                                                 np.log(rangee[0]),
                                                 np.log(rangee[1]))}
            else:
                element = {key: scope.int(dict_type[typee](key,
                                                           np.log(rangee[0]),
                                                           np.log(rangee[1]),
                                                           step))}
        # choice type hyperparameters have different syntax
        elif typee in 'choice':
            element = {key: dict_type[typee](key, [item for item in rangee])}

        # conditional type hyperparameters have different syntax
        elif typee in 'conditional':
            stream_list = []
            for item in rangee:
                stream_dict = {}
                stream_dict.update({key: item['value']})

                for key_item, iteem in item['cond_params'].items():
                    # hyperopt space creator doesn't support repeated labels,
                    # so a ! character is appended to each repeated label
                    while key_item in keys_list:
                        key_item = key_item + '!'
                    keys_list.append(key_item)
                    # generate element by using recursion
                    aux = aux_hyperopt(key_item,
                                       iteem['type'],
                                       iteem['range'],
                                       keys_list,
                                       iteem.get('step', 1))
                    stream_dict.update(aux[0])
                    keys_list = aux[1]

                stream_list.append(stream_dict)
            element = {key: dict_type[typee](key, stream_list)}

        return element, keys_list

    space = {}
    keys_list = []
    # loop over the hyperparameters to optimize dictionary and add each
    # hyperparameter to the space
    for key, item in hyper_to_opt.items():
        # get single hyperparameter
        aux = aux_hyperopt(key,
                           item['type'],
                           item['range'],
                           keys_list,
                           item.get('step', 1))
        # add hyperparameter to the whole hyperparameter space
        space.update(aux[0])
        # update keys_list
        keys_list = aux[1]

    return space
