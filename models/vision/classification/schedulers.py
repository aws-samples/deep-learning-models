import tensorflow as tf


class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Wraps another learning rate scheduler to add a linear or exponential warmup
    """
    
    def __init__(self, optimizer, initial_learning_rate, warmup_steps, warmup_type='linear',
                 dtype=tf.float32):
        super(WarmupScheduler, self).__init__()
        self.optimizer = optimizer
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype)
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.warmup_type = warmup_type
        self.dtype = dtype
        self.optimizer_learning_rate = optimizer(0)
        
    def compute_linear_warmup(self, step):
        return ((self.optimizer_learning_rate*step) + (self.initial_learning_rate*(self.warmup_steps-step)))/self.warmup_steps
    
    @tf.function(experimental_relax_shapes=True)
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if global_step_recomp>=self.warmup_steps:
            return self.optimizer(global_step_recomp - self.warmup_steps)
        return self.compute_linear_warmup(global_step_recomp)
    
    def get_config(self):
        optimizer_config = self.optimizer.get_config()
        optimizer_config['initial_learning_rate'] = self.initial_learning_rate
        optimizer_config['warmup_steps'] = self.warmup_steps
        optimizer_config['warmup_type'] = self.warmup_type


