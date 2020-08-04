import tensorflow as tf

layers = tf.keras.layers


class ConvModule(layers.Layer):
    """
    Module that combines convolutional layer, norm layer, and activation
    Order of layers is currently set to conv, norm, act
    """
    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride,
                 padding='same',
                 use_bias=False,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(2.0, mode='fan_out'),
                 weight_decay=1e-4,
                 norm_cfg=None,
                 act_cfg=None,
                 name=None):
        super(ConvModule, self).__init__()
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size,
            strides=stride,
            use_bias=use_bias,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name=name + '_conv' if name else None)
        self.norm = None
        if norm_cfg and norm_cfg['type'] == 'BN':
            bn_axis = norm_cfg.get('axis', -1)
            eps = norm_cfg.get('eps', 1e-5)
            momentum = norm_cfg.get('momentum', 0.997)
            gamma_initializer = norm_cfg.get('gamma_init', 'ones')
            self.norm = layers.BatchNormalization(axis=bn_axis,
                                                  epsilon=eps,
                                                  gamma_initializer=gamma_initializer,
                                                  momentum=momentum,
                                                  name=name + '_bn')
        self.act = None
        if act_cfg:
            self.act = layers.Activation(act_cfg['type'],
                                         name=name +
                                         '_{}'.format(act_cfg['type']))


    def call(self, x, training=None):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x, training=training)
        if self.act:
            x = self.act(x)
        return x

