import keras


def set_weight_decay(model, alpha):
    if alpha == 0:
        return

    for layer in model.layers:
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            layer.add_loss(keras.regularizers.l2(alpha)(layer.depthwise_kernel))
        elif isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
            layer.add_loss(keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(keras.regularizers.l2(alpha)(layer.bias))
