from ray.tune.automl import ContinuousSpace, DiscreteSpace, SearchSpace


def gen_al_space(self):
    """Create space of hyperparameters for the genetic algorithm optimizer.

    This function creates the space of hyperparameter following ray.tune.automl
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
        ray.tune.automl.search_space.SearchSpace: space of hyperparameters
        following the syntax required by the genetic algorithm optimizer.

    Example::

        hyper_top_opt = {
            'cnn_rnn_dropout':{
                'type': 'uniform',
                'range': [0,1]},
            'optimizer_type':{
                'type': 'choice',,
                'range': ['Adadelta', 'Adam', 'RMSProp', 'SGD']},
            'layer1_filters':{
                'type': 'quniform',
                'range': [16, 64],
                'step': 1}}

    Raises:
        KeyError: if ``type`` is other than ``uniform``, ``quniform`` or
            ``choice``.
    """

    space = []
    # loop over the hyperparameters to optimize dictionary and add each
    # hyperparameter to the space
    for key, item in self.hyperparams_to_optimize.items():
        if item['type'] == 'uniform':
            space.append(ContinuousSpace(
                key,
                item['range'][0],
                item['range'][1],
                (item['range'][0] - item['range'][1])*100))
        elif item['type'] == 'quniform':
            space.append(DiscreteSpace(
                key,
                list(range(item['range'][0],
                           item['range'][1] + item['step'],
                           item['step']))))
        elif item['type'] == 'choice':
            space.append(DiscreteSpace(key,
                                       item['range']))
        else:
            raise KeyError('Genetic algorithm optimization only supports \
                            uniform, quniform and choice space types')
    return SearchSpace(space)
