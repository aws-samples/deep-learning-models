# Copyright (c) Open-MMLab. All rights reserved.

import tensorflow as tf
from .hook import Hook
from ..dist_utils import get_distributed_tape, broadcast_weights


class OptimizerHook(Hook):
    def __init__(self, amp_enabled=False, grad_clip=None):
        self.grad_clip = grad_clip
        self.amp_enabled = amp_enabled

    def after_train_iter(self, runner):
        var_list = runner.model.trainable_variables
        tape = get_distributed_tape(runner.tape) if runner.world_size > 1 else runner.tape
        loss = runner.outputs['loss']
        grads = tape.gradient(loss, var_list)
        # print('RANK {} ITER {} LOSS(after grad calc):'.format(runner.rank, runner.iter), runner.outputs['loss'].numpy())
        grads = [
            grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)
        ]   
        # if self.amp_enabled:
        #    grads = runner.optimizer.get_unscaled_gradients(grads)

        # DEBUG
        # grads = zero_grads = [tf.zeros_like(var) for var in var_list]
        # with open('rank_vars_{}'.format(runner.rank), 'a+') as f:
        #     print('LOSS:', runner.outputs['loss'].numpy(), file=f)
        #     for g, v in zip(grads, var_list):
        #         print(runner.iter, v.name, g.shape, file=f)
        # print('Applying grads at', runner.rank)

        runner.optimizer.apply_gradients(zip(grads, var_list))
        # print('Rank {} applied gradients'.format(runner.rank))
        if runner.epoch == 0 and runner.iter == 0: # broadcast once
            broadcast_weights(runner)


