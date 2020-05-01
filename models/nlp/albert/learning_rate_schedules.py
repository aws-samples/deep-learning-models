import tensorflow as tf


class LinearWarmupLinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_learning_rate, end_learning_rate, warmup_steps, total_steps):
        super().__init__()
        self.max_learning_rate = max_learning_rate
        self.end_learning_rate = end_learning_rate
        self.warmup_steps = warmup_steps
        self.cooldown_steps = total_steps - warmup_steps
        self.total_steps = total_steps

    @tf.function
    def __call__(self, step):
        if step <= self.warmup_steps:
            return self.max_learning_rate * step / float(max(1, self.warmup_steps))
        else:
            return self.end_learning_rate + (
                (self.max_learning_rate - self.end_learning_rate)
                * (self.cooldown_steps - (step - self.warmup_steps))
                / float(max(1, self.cooldown_steps))
            )

    def get_config(self):
        return {
            "max_learning_rate": self.max_learning_rate,
            "end_learning_rate": self.end_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }
