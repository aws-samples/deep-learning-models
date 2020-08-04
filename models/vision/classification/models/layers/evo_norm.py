import tensorflow as tf


class EvoNorm2dS0(tf.keras.layers.Layer):

    def __init__(self, channels, groups=32, eps=1e-5, nonlinear=True):
        super(EvoNorm2dS0, self).__init__()
        self.groups = groups
        self.eps = eps
        self.nonlinear = nonlinear
        self.gamma = self.add_weight(name="gamma", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())
        self.beta = self.add_weight(name="beta", shape=(1, 1, 1, channels), initializer=tf.initializers.Zeros())
        self.v = self.add_weight(name="v", shape=(1, 1, 1, channels), initializer=tf.initializers.Ones())

    def _group_std(self, x):
        input_shape = tf.shape(x)
        N = input_shape[0]
        H = input_shape[1]
        W = input_shape[2]
        C = input_shape[3]
        num_groups = C // self.groups
        x = tf.reshape(x, [N, H, W, self.groups, num_groups])

        _, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        std = tf.sqrt(var + self.eps)
        std = tf.broadcast_to(std, [N, H, W, self.groups, num_groups])
        return tf.reshape(std, input_shape)

    """
    def build(self, input_shape):
        in_channels = input_shape[3]
        ones_init = tf.ones_initializer()
        zeros_init = tf.zeros_initializer()
        self.gamma = tf.Variable(initial_value=ones_init(shape=[1, 1, 1, in_channels]), name='gamma')
        self.beta = tf.Variable(initial_value=zeros_init(shape=[1, 1, 1, in_channels]), name='beta')
        if self.nonlinear:
            self.v = tf.Variable(initial_value=ones_init(shape=[1, 1, 1, in_channels]), name='v')
    """

    @tf.function
    def call(self, x):
        if self.nonlinear:
            num = x * tf.nn.sigmoid(self.v * x)
            return num / self._group_std(x) * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta

if __name__ == "__main__":
    # test
    test_in = tf.random.uniform([256, 40, 40,128])
    out = EvoNorm2dS0(groups=32)(test_in)
