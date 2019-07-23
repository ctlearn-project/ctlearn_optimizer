import numpy as np
from hyperopt import hp, STATUS_OK
from hyperopt.pyll.base import scope
import ctlearn_optimizer.common as common


def hyperopt_space(self):
    """Create hyperopt style hyperparameters space

    Args:
        self

    Returns:
        hyperopt style hyperparameters space to be fed into hyperopt fmin
    """

    def aux_hyperopt(key, typee, rangee, keys_list, step=None,):
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

        if typee in ('uniform', 'quniform', 'normal', 'qnormal'):
            if typee in ('uniform', 'normal'):
                element = {key: dict_type[typee](key,
                                                 rangee[0],
                                                 rangee[1])}
            else:
                element = {key: scope.int(dict_type[typee](key,
                                                           rangee[0],
                                                           rangee[1],
                                                           step))}

        elif typee in ('loguniform', 'qloguniform', 'lognormal', 'qlognormal'):
            if typee in ('loguniform', 'lognormal'):
                element = {key: dict_type[typee](key,
                                                 np.log(rangee[0]),
                                                 np.log(rangee[1]))}
            else:
                element = {key: scope.int(dict_type[typee](key,
                                                           np.log(rangee[0]),
                                                           np.log(rangee[1]),
                                                           step))}

        elif typee in 'choice':
            element = {key: dict_type[typee](key, [item for item in rangee])}

        # type is conditional
        else:
            stream_list = []
            for item in rangee:
                stream_dict = {}
                stream_dict.update({key: item['value']})

                for key_item, iteem in item['cond_params'].items():
                    # append a ! character to repeated keys
                    while key_item in keys_list:
                        key_item = key_item + '!'
                    keys_list.append(key_item)

                    if 'step' in iteem:
                        aux = aux_hyperopt(key_item,
                                           iteem['type'],
                                           iteem['range'],
                                           keys_list,
                                           iteem['step'])
                        stream_dict.update(aux[0])
                        keys_list = aux[1]
                    else:
                        aux = aux_hyperopt(key_item,
                                           iteem['type'],
                                           iteem['range'],
                                           keys_list)
                        stream_dict.update(aux[0])
                        keys_list = aux[1]
                stream_list.append(stream_dict)
            element = {key: dict_type[typee](key, stream_list)}

        return element, keys_list

    params = self.opt_config['Hyperparameters']['Hyperparameters_to_optimize']
    if params is None:
        raise KeyError('Hyperparameters_to_optimize is empty')
    space = {}
    keys_list = []
    for key, item in params.items():
        if 'step' in item:
            aux = aux_hyperopt(key,
                               item['type'],
                               item['range'],
                               keys_list,
                               item['step'])
            space.update(aux[0])
            keys_list = aux[1]
        else:
            aux = aux_hyperopt(key,
                               item['type'],
                               item['range'],
                               keys_list)
            space.update(aux[0])
            keys_list = aux[1]
    return space


def objective(self, hyperparams):
    """Objective function for hyperopt input - output workflow

    Args:
        self
        hyperparams: set of hyperparameters to evaluate provided by
                     hyperopt fmin [dict]

    Returns:
        metric to minimize by hyperopt fmin [dict]
    """
    loss = common.objective(self, hyperparams)
    return {'loss': loss, 'status': STATUS_OK}
