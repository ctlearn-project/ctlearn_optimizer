import skopt
import ctlearn_optimizer.common as common


def skopt_space(self):
    """Create skopt style hyperparameters space

    Args:
        self

    Returns:
        skopt style hyperparameters space to be fed into skopt minimize function
    """

    params = self.opt_config['Hyperparameters']['Hyperparameters_to_optimize']
    space = []
    for key, items in params.items():
        if items['type'] == 'uniform':
            space.append(skopt.space.Real(
                items['range'][0], items['range'][1], name=key))
        elif items['type'] == 'quniform':
            space.append(skopt.space.Integer(
                items['range'][0], items['range'][1], name=key))
        elif items['type'] == 'loguniform':
            space.append(skopt.space.Real(
                items['range'][0], items['range'][1], name=key,
                prior='log-uniform'))
        elif items['type'] == 'choice':
            space.append(skopt.space.Categorical(items['range'], name=key))
        else:
            raise KeyError('Gaussian Processes based BO only supports uniform,\
                           quniform, loguniform and choice space types')
    return space


def objective(self, hyperparams):
    """Objective function for skopt input - output workflow

    Args:
        self
        hyperparams: set of hyperparameters to evaluate provided by
                     skopt optimizer [dict]

    Returns:
        metric to minimize by skopt optimizer [float]
    """
    # skopt returns numpy scalars instead of plain values for the hyperparameters,
    # so we have to convert each numpy scalar back into a plain Python float or int
    for key, value in hyperparams.items():
        if not isinstance(value, str):
            hyperparams[key] = value.tolist()

    loss = common.objective(self, hyperparams)
    return loss
