from keras.optimizers import SGD, Adam, RMSprop


def set_optimizer(optimizer, lr, mom=0.):
    assert optimizer in ['sgd', 'adam', 'rmsprop']
    if optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=mom, clipnorm=1.)
    elif optimizer == 'adam':
        opt = Adam(lr=lr, beta_1=mom, clipnorm=1.)
    else:
        opt = RMSprop(lr=lr, clipnorm=1.)

    return opt
