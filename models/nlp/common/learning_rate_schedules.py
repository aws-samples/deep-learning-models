import tensorflow as tf


class LinearWarmupPolyDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_learning_rate, end_learning_rate, warmup_steps, total_steps, power=1.0):
        super().__init__()
        self.max_learning_rate = max_learning_rate
        self.end_learning_rate = end_learning_rate
        self.warmup_steps = warmup_steps
        self.cooldown_steps = total_steps - warmup_steps
        self.total_steps = total_steps
        self.power = power

    @tf.function
    def __call__(self, step):
        if step <= self.warmup_steps:
            return self.max_learning_rate * step / float(max(1, self.warmup_steps))
        else:
            cooldown_pct_left = (self.total_steps - step) / float(max(1, self.cooldown_steps))
            return (
                (self.max_learning_rate - self.end_learning_rate)
                * (cooldown_pct_left ** self.power)
            ) + self.end_learning_rate

    def get_config(self):
        return {
            "max_learning_rate": self.max_learning_rate,
            "end_learning_rate": self.end_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }
