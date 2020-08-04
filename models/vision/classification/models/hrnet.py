"""
HRNetV2 - High Resolution Net model implementation in TensorFlow 2

# Reference

- [Deep High-Resolution Representation Learning for Visual Recognition] (https://arxiv.org/pdf/1908.07919.pdf)

# Reference implementations

- [HRNet] https://github.com/HRNet/HRNet-Image-Classification
- [hrnet-tf] https://github.com/yuanyuanli85/tf-hrnet
"""

import tensorflow as tf
from .layers.conv_module import ConvModule


layers = tf.keras.layers


class BasicBlock(layers.Layer):
    def __init__(self,
                 channels,
                 norm_cfg,
                 act_cfg,
                 expansion=1,
                 weight_decay=1e-4,
                 name=None,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv_mod1 = ConvModule(channels,
                                    3,
                                    stride=stride,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name=name)

        self.conv_mod2 = ConvModule(channels,
                                    3,
                                    stride=1,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=None,
                                    name=name)

        self.downsample = downsample


    def call(self, x, training=None):
        residual = x
        x = self.conv_mod1(x, training=training)
        x = self.conv_mod2(x, training=training)

        if self.downsample:
            residual = self.downsample(residual, training=training)

        x = x + residual
        x = tf.nn.relu(x)
        return x


class Bottleneck(layers.Layer):
    def __init__(self,
                 channels,
                 norm_cfg,
                 act_cfg,
                 expansion=4,
                 weight_decay=1e-4,
                 stride=1,
                 downsample=None,
                 name=None):
        super(Bottleneck, self).__init__()
        self.conv_mod1 = ConvModule(channels,
                                    1,
                                    stride=1,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name=name)

        self.conv_mod2 = ConvModule(channels,
                                    3,
                                    stride=stride,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name=name)

        self.conv_mod3 = ConvModule(channels * expansion,
                                    1,
                                    stride=1,
                                    padding='same',
                                    use_bias=False,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=None,
                                    name=name)

        self.downsample = downsample

    def call(self, x, training=None):
        residual = x
        x = self.conv_mod1(x, training=training)
        x = self.conv_mod2(x, training=training)
        x = self.conv_mod3(x, training=training)

        if self.downsample:
            residual = self.downsample(residual, training=training)

        x = x + residual
        x = tf.nn.relu(x)
        return x


class HRModule(layers.Layer):
    def __init__(self, cfg, module_idx, multiscale_output=True):
        super(HRModule, self).__init__()
        self.stage_name = cfg['name']
        self.module_idx = module_idx
        self.weight_decay = cfg.get('weight_decay', 1e-4)
        self.norm_cfg = cfg.get('norm_cfg', None)
        self.act_cfg = cfg.get('act_cfg', None)
        self.num_branches = cfg['num_branches']
        self.num_blocks = cfg['num_blocks']
        self.num_channels = cfg['num_channels']
        assert self.num_branches == len(self.num_blocks)
        assert self.num_branches == len(self.num_channels)
        self.multiscale_output = multiscale_output
        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()


    def _make_branch(self, branch_level):
        blocks = []
        branch_name = 'hrm_{}_module{}_branch{}'.format(self.stage_name, self.module_idx, branch_level)
        for block_index in range(self.num_blocks[branch_level]):
            block_name = '{}_block{}'.format(branch_name, block_index)
            blocks.append(
                BasicBlock(self.num_channels[branch_level],
                           self.norm_cfg,
                           self.act_cfg,
                           self.weight_decay,
                           name=block_name))
        return tf.keras.Sequential(blocks, name=branch_name)


    def _make_branches(self):
        branches = []
        for i in range(self.num_branches):
            branches.append(self._make_branch(i))
        return branches


    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        fuse_layers = []
        for i in range(self.num_branches if self.multiscale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:  # upsample low to high
                    conv_mod = ConvModule(self.num_channels[i],
                                          1,
                                          stride=1,
                                          padding='same',
                                          use_bias=False,
                                          weight_decay=self.weight_decay,
                                          norm_cfg=self.norm_cfg,
                                          act_cfg=self.act_cfg,
                                          name='fuse_{}_{}_{}'.format(
                                              self.stage_name, j, i))
                    upsample = layers.UpSampling2D(size=(2**(j-i),2**(j-i)), interpolation='nearest')
                    fuse_layer.append(tf.keras.Sequential([conv_mod, upsample]))
                elif j == i:
                    fuse_layer.append(None)
                else:  # downsample 3x3 stride 2
                    down_layers = []
                    for k in range(i-j):
                        if k == i-j-1:
                            down_name = 'fusedown_{}_{}_{}'.format(self.stage_name, j, i)
                            conv_mod = ConvModule(
                                self.num_channels[i],
                                3,
                                stride=2,
                                padding='same',
                                use_bias=False,
                                weight_decay=self.weight_decay,
                                norm_cfg=self.norm_cfg,
                                act_cfg=None,
                                name=down_name)
                        else:
                            conv_mod = ConvModule(
                                self.num_channels[j],
                                3,
                                stride=2,
                                padding='same',
                                use_bias=False,
                                weight_decay=self.weight_decay,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg,
                                name=down_name)
                        down_layers.append(conv_mod)
                    fuse_layer.append(tf.keras.Sequential(down_layers))
            fuse_layers.append(fuse_layer)
        return fuse_layers


    def call(self, x, training=None):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i], training=training)
        if self.num_branches == 1:
            return x
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0]
            if i > 0:
                y = self.fuse_layers[i][0](x[0], training=training)
            for j in range(1, self.num_branches):
                if i == j: # None layer case
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j], training=training)
            x_fuse.append(tf.nn.relu(y))

        return x_fuse


class Stem(layers.Layer):
    def __init__(self, cfg):
        super(Stem, self).__init__()
        filters = cfg['channels']
        kernel_size = cfg['kernel_size']
        stride = cfg['stride']
        use_bias = cfg['use_bias']
        padding = cfg['padding']
        weight_decay = cfg['weight_decay']

        self.conv_mod1 = ConvModule(filters,
                                    kernel_size,
                                    stride,
                                    padding=padding,
                                    use_bias=use_bias,
                                    weight_decay=weight_decay,
                                    norm_cfg=cfg.get('norm_cfg', None),
                                    act_cfg=cfg.get('act_cfg', None),
                                    name='stem_1')

        self.conv_mod2 = ConvModule(filters,
                                    kernel_size,
                                    stride,
                                    padding=padding,
                                    use_bias=use_bias,
                                    weight_decay=weight_decay,
                                    norm_cfg=cfg.get('norm_cfg', None),
                                    act_cfg=cfg.get('act_cfg', None),
                                    name='stem_2')

    def call(self, x, training=None):
        out = self.conv_mod1(x, training=training)
        out = self.conv_mod2(out, training=training)
        return out


class Transition(layers.Layer):
    def __init__(self, cfg, prev_layer_branches, prev_layer_channels, name=None):
        super(Transition, self).__init__(name=name)
        wd = cfg['weight_decay']
        norm_cfg = cfg.get('norm_cfg', None)
        act_cfg = cfg.get('act_cfg', None)
        self.num_branches = cfg['num_branches']
        curr_stage_channels = cfg['num_channels']
        self.transition_layers = []
        for i in range(self.num_branches):
            if i < prev_layer_branches:
                if prev_layer_channels[i] != curr_stage_channels[i]:
                    convmod = ConvModule(curr_stage_channels[i],
                                         3,
                                         1,
                                         padding='same',
                                         use_bias=False,
                                         weight_decay=wd,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg,
                                         name='transition_{}_{}'.format(
                                             len(curr_stage_channels) - 1,
                                             i + 1))
                    self.transition_layers.append(convmod)
                else:
                    self.transition_layers.append(None) # pass input as is
            else:
                # this handles the new branch(es) in the current stage
                new_transitions = []
                for j in range(i + 1 - prev_layer_branches):
                    if j == i - prev_layer_branches:
                        channels = curr_stage_channels[i]
                    else:
                        channels = prev_layer_channels[-1]
                    convmod = ConvModule(channels,
                                         3,
                                         2,
                                         padding='same',
                                         use_bias=False,
                                         weight_decay=wd,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg,
                                         name='new_transition_{}_{}'.format(
                                             len(curr_stage_channels) - 1,
                                             i + 1))
                    new_transitions.append(convmod)
                self.transition_layers.append(tf.keras.Sequential(new_transitions))


    def call(self, x, training=None):
        outputs = []
        for i, tl in enumerate(self.transition_layers):
            if tl:
                transition = tl(x[-1], training=training)
                outputs.append(transition)
            else:
                outputs.append(x[i])
        return outputs


class Front(layers.Layer):
    def __init__(self, cfg, expansion=4):
        super(Front, self).__init__(name=cfg['name'])
        wd = cfg['weight_decay']
        norm_cfg = cfg.get('norm_cfg', None)
        act_cfg = cfg.get('act_cfg', None)
        num_blocks = cfg['num_blocks'][0]
        channels_list = cfg['num_channels']
        channels = channels_list[0]
        downsample = ConvModule(channels * expansion,
                                1,
                                1,
                                padding='same',
                                use_bias=False,
                                weight_decay=wd,
                                norm_cfg=norm_cfg,
                                act_cfg=None,
                                name="front_downsample")
        # block = residual unit
        self.blocks = []
        self.blocks.append(
            Bottleneck(channels,
                       norm_cfg,
                       act_cfg,
                       name='front_bottleneck_1',
                       weight_decay=wd,
                       stride=1,
                       downsample=downsample))
        for i in range(1, num_blocks):
            self.blocks.append(
                Bottleneck(channels,
                           norm_cfg,
                           act_cfg,
                           name='front_bottleneck_{}'.format(i + 1),
                           weight_decay=wd,
                           stride=1,
                           downsample=None))
        self.stage = tf.keras.Sequential(self.blocks)

    def call(self, x, training=None):
        return self.stage(x, training=training)


class BottleneckStage(layers.Layer):
    def __init__(self,
                 channels,
                 num_blocks,
                 expansion=4,
                 stride=1,
                 weight_decay=1e-4,
                 norm_cfg=None,
                 act_cfg=None):
        super(BottleneckStage, self).__init__()
        downsample = ConvModule(channels * expansion,
                                1,
                                1,
                                padding='same',
                                use_bias=False,
                                weight_decay=weight_decay,
                                norm_cfg=norm_cfg,
                                act_cfg=None,
                                name="cls_downsample")
        self.blocks = []
        self.blocks.append(
            Bottleneck(channels,
                       norm_cfg,
                       act_cfg,
                       name='cls_bottleneck_1',
                       weight_decay=weight_decay,
                       stride=1,
                       downsample=downsample))
        for i in range(1, num_blocks):
            self.blocks.append(
                Bottleneck(channels,
                           norm_cfg,
                           act_cfg,
                           name='cls_bottleneck_{}'.format(i + 1),
                           weight_decay=weight_decay,
                           stride=1,
                           downsample=None))
        self.stage = tf.keras.Sequential(self.blocks)

    def call(self, x, training=None):
        return self.stage(x, training=training)


class Stage(layers.Layer):
    def __init__(self, cfg, multiscale_output=True):
        super(Stage, self).__init__(name=cfg['name'])
        self.num_modules = cfg['num_modules']
        self.num_branches = cfg['num_branches']
        self.modules = []
        for module_idx in range(self.num_modules):
            if not multiscale_output and module_idx == self.num_modules-1:
                multiscale_output = False
            else:
                multiscale_output = True

            hr_module = HRModule(
                cfg, module_idx, multiscale_output=multiscale_output)
            self.modules.append(hr_module)


    def call(self, x_list, training=None):
        out = x_list
        for module in self.modules:
            out = module(out, training=training)
        return out


class ClsHead(layers.Layer):
    def __init__(self, cfg, expansion=4):
        super(ClsHead, self).__init__()
        channels = cfg['channels']
        weight_decay = cfg.get('weight_decay', 1e-4)
        norm_cfg = cfg.get('norm_cfg', None)
        act_cfg = cfg.get('act_cfg', None)
        num_classes = cfg.get('num_classes', 1000)
        fc_channels = cfg.get('fc_channels', 2048)
        # C, 2C, 4C, 8C -> 128, 256, 512, 1024
        self.width_incr_layers = []
        for i in range(len(channels)):
            incr_layer = BottleneckStage(channels[i],
                                         1,
                                         stride=1,
                                         weight_decay=weight_decay,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg)
            self.width_incr_layers.append(incr_layer)
        # downsampling layers
        self.downsample_layers = []
        for i in range(1, len(channels)):
            downsample = ConvModule(channels[i] * expansion,
                                    3,
                                    2,
                                    padding='same',
                                    use_bias=True,
                                    weight_decay=weight_decay,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    name='downsample_cls_{}'.format(i))
            self.downsample_layers.append(downsample)
        self.final_layer = ConvModule(fc_channels,
                                      1,
                                      1,
                                      padding='same',
                                      use_bias=True,
                                      weight_decay=weight_decay,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg,
                                      name='final_{}'.format(i))
        self.classifier = layers.Dense(num_classes,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='logits')


    def call(self, x_list, training=None):
        y = self.width_incr_layers[0](x_list[0], training=training)
        for i in range(1, len(self.downsample_layers)+1):
            y = self.width_incr_layers[i](x_list[i], training=training) + self.downsample_layers[i-1](y, training=training)

        y = self.final_layer(y)
        y = layers.AveragePooling2D(pool_size=(7, 7), strides=1)(y)
        y = layers.Flatten()(y)
        y = self.classifier(y)
        # y = layers.Activation('softmax', dtype='float32')(y)
        return y


class HRNet(tf.keras.Model):
    def __init__(self, model_cfg):
        super(HRNet, self).__init__()

        # stem
        self.stem = Stem(model_cfg['stem'])
        # stages
        self.stages = []
        self.transitions = []
        for s in range(1, model_cfg['num_stages'] + 1):
            stage_cfg = model_cfg['stage{}'.format(s)]
            if s == 1:
                # bottleneck units
                self.stages.append(Front(stage_cfg))
            else:
                # basic units
                prev_stage_cfg = model_cfg['stage{}'.format(s - 1)]
                prev_layer_branches = prev_stage_cfg['num_branches']
                prev_layer_channels = [prev_stage_cfg['expansion'] * c for c in prev_stage_cfg['num_channels']]
                trans_name = 'transition_{}_{}'.format(s-1, s)
                self.transitions.append(Transition(stage_cfg, prev_layer_branches, prev_layer_channels, name=trans_name))
                self.stages.append(Stage(stage_cfg))

        # classification head
        head_cfg = model_cfg['head']
        self.cls_head = ClsHead(head_cfg)


    def call(self, x, training=None):
        t0 = tf.timestamp()
        x = self.stem(x, training=training)
        t1 = tf.timestamp()
        front = self.stages[0]
        stage1_output = front(x, training=training)
        t2 = tf.timestamp()
        transition12 = self.transitions[0]
        stage2 = self.stages[1]
        stage1_transitions = transition12([stage1_output], training=training)
        t3 = tf.timestamp()
        stage2_outputs = stage2(stage1_transitions, training=training)
        t4 = tf.timestamp()
        transition23 = self.transitions[1]
        stage3 = self.stages[2]
        stage2_transitions = transition23(stage2_outputs, training=training)
        t5 = tf.timestamp()
        stage3_outputs = stage3(stage2_transitions, training=training)
        t6 = tf.timestamp()
        transition34 = self.transitions[2]
        stage4 = self.stages[3]
        stage3_transitions = transition34(stage3_outputs, training=training)
        t7 = tf.timestamp()
        stage4_outputs = stage4(stage3_transitions, training=training)
        t8 = tf.timestamp()
        # classification
        y = self.cls_head(stage4_outputs)
        t9 = tf.timestamp()
        """
        tf.print('stem', t1-t0)
        tf.print('stage1', t2-t1)
        tf.print('trans12', t3-t2)
        tf.print('stage2', t4-t3)
        tf.print('trans23', t5-t4)
        tf.print('stage3', t6-t5)
        tf.print('trans34', t7-t6)
        tf.print('stage4', t8-t7)
        tf.print('head', t9-t8)
        """
        return y


def build_hrnet(model_name):
    # CONFIG W32C
    model_w32c = dict(type='HRNet',
                 num_stages=4,
                 stem=dict(
                     channels=64,
                     kernel_size=3,
                     stride=2,
                     padding='same',
                     use_bias=False,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage1=dict(
                     name='s1',
                     num_modules=1,
                     num_branches=1,
                     num_blocks=(4, ),
                     num_channels=(64, ),
                     expansion = 4,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage2=dict(
                     name='s2',
                     num_modules=1,
                     num_branches=2,
                     num_blocks=(4, 4),
                     num_channels=(32, 64),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage3=dict(
                     name='s3',
                     num_modules=4,
                     num_branches=3,
                     num_blocks=(4, 4, 4),
                     num_channels=(32, 64, 128),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage4=dict(
                     name='s4',
                     num_modules=3,
                     num_branches=4,
                     num_blocks=(4, 4, 4, 4),
                     num_channels=(32, 64, 128, 256),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 head=dict(
                     name='cls_head',
                     channels=(32, 64, 128, 256),
                     expansion = 4,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ))

    # CONFIG W18C
    model_w18c = dict(type='HRNet',
                 num_stages=4,
                 stem=dict(
                     channels=64,
                     kernel_size=3,
                     stride=2,
                     padding='same',
                     use_bias=False,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage1=dict(
                     name='s1',
                     num_modules=1,
                     num_branches=1,
                     num_blocks=(4, ),
                     num_channels=(64, ),
                     expansion = 4,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage2=dict(
                     name='s2',
                     num_modules=1,
                     num_branches=2,
                     num_blocks=(4, 4),
                     num_channels=(18, 36),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage3=dict(
                     name='s3',
                     num_modules=4,
                     num_branches=3,
                     num_blocks=(4, 4, 4),
                     num_channels=(18, 36, 72),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 stage4=dict(
                     name='s4',
                     num_modules=3,
                     num_branches=4,
                     num_blocks=(4, 4, 4, 4),
                     num_channels=(18, 36, 72, 144),
                     expansion = 1,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ),
                 head=dict(
                     name='cls_head',
                     channels=(32, 64, 128, 256),
                     expansion = 4,
                     act_cfg=dict(type='relu', ),
                     norm_cfg=dict(
                         type='BN',
                         axis=-1,
                         momentum=0.9,
                         eps=1e-5,
                     ),
                     weight_decay=5e-5,
                 ))



    train_cfg = dict(weight_decay=5e-5, )
    dataset_type = 'imagenet'
    dataset_mean = ()
    dataset_std = ()
    data_root = '/data/imagenet'
    data = dict(
        imgs_per_gpu=128,
        train=dict(
            type=dataset_type,
            train=True,
            dataset_dir=data_root,
            tf_record_pattern='train-*',
            resize_dim=256,
            crop_dim=224,
            augment=True,
            mean=(),
            std=(),
        ),
        val=dict(
            type=dataset_type,
            train=False,
            dataset_dir=data_root,
            tf_record_pattern='val-*',
            resize_dim=256,
            crop_dim=224,
            augment=False,
            mean=(),
            std=(),
        ),
    )
    evaluation = dict(interval=1)
    # optimizer
    optimizer = dict(
        type='SGD',
        learning_rate=1e-2,
        momentum=0.9,
        nesterov=True,
    )
    # extra options related to optimizers
    optimizer_config = dict(amp_enabled=True, )
    # learning policy
    lr_config = dict(policy='step',
                     warmup='linear',
                     warmup_epochs=5,
                     warmup_ratio=1.0 / 3,
                     step=[30, 60, 90])


    checkpoint_config = dict(interval=1, outdir='checkpoints')
    log_config = dict(interval=50, )
    total_epochs = 100,
    log_level = 'INFO'
    work_dir = './work_dirs/hrnet_w32_cls'
    resume_from = None

    if model_name == 'hrnet_w18c':
        hrnet = HRNet(model_w18c)
    elif model_name == 'hrnet_w32c':
        hrnet = HRNet(model_w32c)

    # pass dummy data to init network
    x = tf.random.uniform([8, 224, 224, 3])
    _ = hrnet(x)
    # hrnet.summary()
    return hrnet

if __name__ == "__main__":
    build_hrnet(model_name)

