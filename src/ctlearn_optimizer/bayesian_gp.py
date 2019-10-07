import skopt


def skopt_space(hyper_to_opt):
    """Create space of hyperparameters for the gaussian processes optimizer.

    This function creates the space of hyperparameter following skopt syntax.

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
        list: space of hyperparameters following the syntax required by the
        gaussian processes optimization algorithm.

    Example::

        hyper_top_opt = {
            'cnn_rnn_dropout':{
                'type': 'uniform',
                'range': [0,1]},
            'optimizer_type':{
                'type': 'choice',,
                'range': ['Adadelta', 'Adam', 'RMSProp', 'SGD']},
            'base_learning_rate':{
                'type': 'loguniform',
                'range': [-5, 0]},
            'layer1_filters':{
                'type': 'quniform',
                'range': [16, 64],
                'step': 1}}

    Raises:
        KeyError: if ``type`` is other than ``uniform``, ``quniform``,
          ``loguniform`` or ``choice``.
    """

    space = []
    # loop over the hyperparameters to optimize dictionary and add each
    # hyperparameter to the space
    for key, items in hyper_to_opt.items():
        if items['type'] == 'uniform':
            space.append(skopt.space.Real(items['range'][0],
                                          items['range'][1],
                                          name=key))
        elif items['type'] == 'quniform':
            space.append(skopt.space.Integer(items['range'][0],
                                             items['range'][1],
                                             name=key))
        elif items['type'] == 'loguniform':
            space.append(skopt.space.Real(items['range'][0],
                                          items['range'][1],
                                          name=key,
                                          prior='log-uniform'))
        elif items['type'] == 'choice':
            space.append(skopt.space.Categorical(items['range'],
                                                 name=key))
        else:
            raise KeyError('The gaussian processes optimizer supports only \
                uniform, quniform, loguniform and choice space types')
    return space
