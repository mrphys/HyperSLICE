# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Work-in-progress layers."""

import tensorflow as tf
import model.conv_blocks as local_conv_blocks
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import layer_util

@tf.keras.utils.register_keras_serializable(package='Playground')
class TimeUNet(tf.keras.layers.Layer):
  """A TimeUNet layer. (time distributed 2D Unet)
  Args:
    scales: The number of scales. `scales - 1` pooling layers will be added to
      the model. Lowering the depth may reduce the amount of memory required for
      training.
    base_filters: The number of filters that the first layer in the
      convolution network will have. The number of filters in following layers
      will be calculated from this number. Lowering this number may reduce the
      amount of memory required for training.
    kernel_size: An integer or tuple/list of `rank` integers, specifying the
      size of the convolution window. Can be a single integer to specify the
      same value for all spatial dimensions.
    pool_size: The pooling size for the pooling operations. Defaults to 2.
    block_depth: The number of layers in each convolutional block. Defaults to
      2.
    use_deconv: If `True`, transpose convolution (deconvolution) will be used
      instead of up-sampling. This increases the amount memory required during
      training. Defaults to `False`.
    rank: An integer specifying the number of spatial dimensions. Defaults to 2.
    activation: A callable or a Keras activation identifier. Defaults to
      `'relu'`.
    kernel_initializer: A `tf.keras.initializers.Initializer` or a Keras
      initializer identifier. Initializer for convolutional kernels. Defaults to
      `'VarianceScaling'`.
    bias_initializer: A `tf.keras.initializers.Initializer` or a Keras
      initializer identifier. Initializer for bias terms. Defaults to `'Zeros'`.
    kernel_regularizer: A `tf.keras.initializers.Regularizer` or a Keras
      regularizer identifier. Regularizer for convolutional kernels. Defaults to
      `None`.
    bias_regularizer: A `tf.keras.initializers.Regularizer` or a Keras
      regularizer identifier. Regularizer for bias terms. Defaults to `None`.
    use_batch_norm: If `True`, use batch normalization. Defaults to `False`.
    use_sync_bn: If `True`, use synchronised batch normalization. Defaults to
      `False`.
    bn_momentum: A `float`. Momentum for the moving average in batch
      normalization.
    bn_epsilon: A `float`. Small float added to variance to avoid dividing by
      zero during batch normalization.
    out_channels: An `int`. The number of output channels.
    out_activation: A callable or a Keras activation identifier. The output
      activation. Defaults to `None`.
    use_global_residual: A `boolean`. If `True`, adds a global residual
      connection to create a residual learning network. Defaults to `False`.
    use_dropout: A `boolean`. If `True`, a dropout layer is inserted after
      each activation. Defaults to `False`.
    dropout_rate: A `float`. The dropout rate. Only relevant if `use_dropout` is
      `True`. Defaults to 0.3.
    dropout_type: A `str`. The dropout type. Must be one of `'standard'` or
      `'spatial'`. Standard dropout drops individual elements from the feature
      maps, whereas spatial dropout drops entire feature maps. Only relevant if
      `use_dropout` is `True`. Defaults to `'standard'`.
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self,
               scales,
               base_filters,
               kernel_size,
               pool_size=2,
               rank=2,
               block_depth=2,
               use_deconv=False,
               activation='relu',
               kernel_initializer='VarianceScaling',
               bias_initializer='Zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bias=True,
               use_conv_pool=False,
               use_add_merge=False,
               use_batch_norm=False,
               use_sync_bn=False,
               bn_momentum=0.99,
               bn_epsilon=0.001,
               out_channels=None,
               out_activation=None,
               use_global_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               time_distributed=True,
               selected_frame=-1,
               **kwargs):
    """Creates a UNet layer."""
    self._scales = scales
    self._base_filters = base_filters
    self._kernel_size = kernel_size
    self._pool_size = pool_size
    self._rank = rank
    self._block_depth = block_depth
    self._use_deconv = use_deconv
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    #self._recurrent_regularizer = recurrent_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bias=use_bias
    self._use_add_merge=use_add_merge
    self._use_batch_norm = use_batch_norm
    self._use_sync_bn = use_sync_bn
    self._bn_momentum = bn_momentum
    self._bn_epsilon = bn_epsilon
    self._out_channels = out_channels
    self._out_activation = out_activation
    self._use_global_residual = use_global_residual
    self._use_dropout = use_dropout
    self._dropout_rate = dropout_rate
    self._dropout_type = check_util.validate_enum(
        dropout_type, {'standard', 'spatial'}, 'dropout_type')
    self._time_distributed=time_distributed
    self._selected_frame=selected_frame
    self._use_conv_pool=use_conv_pool

    block_config = dict(
        filters=None, # To be filled for each scale.
        kernel_size=self._kernel_size,
        strides=1,
        rank=self._rank,
        activation=self._activation,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_bias=self._use_bias,
        use_batch_norm=self._use_batch_norm,
        use_sync_bn=self._use_sync_bn,
        bn_momentum=self._bn_momentum,
        bn_epsilon=self._bn_epsilon,
        use_dropout=self._use_dropout,
        dropout_rate=self._dropout_rate,
        dropout_type=self._dropout_type,
        time_distributed=self._time_distributed,
        )

    #pool = layer_util.get_nd_layer('MaxPool', self._rank)
    if self._use_conv_pool:
        if self._time_distributed:
          pool = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(
                      layer_util.get_nd_layer('Conv', self._rank)(*args, **kwargs))
        else:
          pool = layer_util.get_nd_layer('Conv', self._rank)
        pool_config = dict(
          filters=None,  # To be filled for each scale.
          kernel_size=self._kernel_size,
          strides=self._pool_size,
          padding='same',
          activation=self._activation,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=None,
          bias_regularizer=self._bias_regularizer,
          use_bias=self._use_bias)
    else:
      if self._time_distributed:
        pool = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(
                      layer_util.get_nd_layer('MaxPool', self._rank)(*args, **kwargs))
      else:
        pool = layer_util.get_nd_layer('MaxPool', self._rank)
    
    if use_deconv is True:

      #upsamp = layer_util.get_nd_layer('ConvTranspose', self._rank)
      if self._time_distributed:
        upsamp  = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(
                    layer_util.get_nd_layer('ConvTranspose', self._rank)(*args, **kwargs))
      else:
        upsamp = layer_util.get_nd_layer('ConvTranspose', self._rank)
      upsamp_config = dict(
          filters=None,  # To be filled for each scale.
          kernel_size=self._kernel_size,
          strides=self._pool_size,
          padding='same',
          activation=None,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=None,
          bias_regularizer=self._bias_regularizer,
          use_bias=self._use_bias)

      
    elif use_deconv=='PixelShuffle':
      upsamp = local_conv_blocks.PixelShuffle
      upsamp_config = dict(
          filters=None,  # To be filled for each scale.
          kernel_size=self._kernel_size,
          pool_size=self._pool_size,
          strides=1,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=None,
          bias_regularizer=self._bias_regularizer,
          use_bias=self._use_bias)

    else:
      if self._time_distributed:
        upsamp  = lambda *args, **kwargs: tf.keras.layers.TimeDistributed(
                    layer_util.get_nd_layer('UpSampling', self._rank)(*args, **kwargs))
      else:
        upsamp = layer_util.get_nd_layer('UpSampling', self._rank)
      upsamp_config = dict(
          size=self._pool_size)

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._channel_axis = -1
    else:
      self._channel_axis = 1

    self._enc_blocks = []
    self._dec_blocks = []
    self._pools = []
    self._upsamps = []
    self._concats = []

    # Configure backbone and decoder.
    for scale in range(self._scales):
      num_filters = base_filters * (2 ** scale)
      block_config['filters'] = [num_filters] * self._block_depth
      self._enc_blocks.append(local_conv_blocks.LocalConvBlock(**block_config))

      if scale < self._scales - 1:
        if self._use_conv_pool:
          pool_config['filters']= num_filters * self._pool_size
          self._pools.append(pool(**pool_config))
        else:
          self._pools.append(pool(
              pool_size=self._pool_size,
              strides=self._pool_size,
              padding='same'))
        
        if use_deconv is True: 
          upsamp_config['filters'] = num_filters
        elif use_deconv=='PixelShuffle':
          upsamp_config['filters'] = num_filters * self._pool_size
        self._upsamps.append(upsamp(**upsamp_config))
        self._concats.append(tf.keras.layers.Concatenate(
            axis=self._channel_axis))
        self._dec_blocks.append(local_conv_blocks.LocalConvBlock(**block_config))

    # Configure output block.
    if self._out_channels is not None:
      block_config['filters'] = self._out_channels
      # If network is residual, the activation is performed after the residual
      # addition.
      if self._use_global_residual:
        block_config['activation'] = None
      else:
        block_config['activation'] = self._out_activation
      self._out_block = local_conv_blocks.LocalConvBlock(**block_config)

    # Configure residual addition, if requested.
    self._add = tf.keras.layers.Add()
    if self._use_global_residual:
      self._out_activation_fn = tf.keras.activations.get(self._out_activation)

    super().__init__(**kwargs)

  def call(self, inputs, training=None): # pylint: disable=missing-param-doc,unused-argument
    """Runs forward pass on the input tensors."""
    x = inputs

    # Backbone.
    cache = [None] * (self._scales - 1) # For skip connections to decoder.
    for scale in range(self._scales - 1):
      cache[scale] = self._enc_blocks[scale](x)
      x = self._pools[scale](cache[scale])
    x = self._enc_blocks[-1](x)

    # Decoder.
    for scale in range(self._scales - 2, -1, -1):
      x = self._upsamps[scale](x)
      if self._use_add_merge:
        x = self._add([x, cache[scale]])
      else:
        x = self._concats[scale]([x, cache[scale]])
      x = self._dec_blocks[scale](x)

    # Head.
    if self._out_channels is not None:
      x = self._out_block(x)

    # Global residual connection.
    if self._use_global_residual:
      x = self._add([x, tf.expand_dims(inputs[:,:,:,self._selected_frame],axis=-1)])
      if self._out_activation is not None:
        x = self._out_activation_fn(x)

    return x

  def get_config(self):
    """Gets layer configuration."""
    config = {
        'scales': self._scales,
        'base_filters': self._base_filters,
        'kernel_size': self._kernel_size,
        'pool_size': self._pool_size,
        'rank': self._rank,
        'block_depth': self._block_depth,
        'use_deconv': self._use_deconv,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_batch_norm': self._use_batch_norm,
        'use_bias': self._use_bias,
        'use_add_merge':self._use_add_merge,
        'use_conv_pool':self._use_conv_pool,
        'use_sync_bn': self._use_sync_bn,
        'bn_momentum': self._bn_momentum,
        'bn_epsilon': self._bn_epsilon,
        'out_channels': self._out_channels,
        'out_activation': self._out_activation,
        'use_global_residual': self._use_global_residual,
        'use_dropout': self._use_dropout,
        'dropout_rate': self._dropout_rate,
        'dropout_type': self._dropout_type,
        'time_distributed':self._time_distributed,
        'selected_frame':self._selected_frame,
        #'stateful': self._stateful,
        #'return_sequences': self._return_sequences,
    }
    base_config = super().get_config()
    return {**base_config, **config}       
    
@tf.keras.utils.register_keras_serializable(package='Playground')
class FastDVDNet(tf.keras.layers.Layer):
  """A FastDVDNet model. 
  Args:
    Keyword arguments of the TimeUNet
    input_format: set to complex for inputs of format (x,y,10 channels(5real+5imag channels))
    **kwargs: Additional keyword arguments to be passed to base class.
  """
  def __init__(self,
               scales,
               base_filters,
               kernel_size,
               pool_size=2,
               rank=2,
               block_depth=2,
               use_deconv=False,
               activation='relu',
               kernel_initializer='VarianceScaling',
               bias_initializer='Zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bias=True,
               use_add_merge=False,
               use_conv_pool=False,
               use_batch_norm=False,
               use_sync_bn=False,
               bn_momentum=0.99,
               bn_epsilon=0.001,
               out_channels=None,
               out_activation=None,
               use_global_residual=False,
               use_dropout=False,
               dropout_rate=0.3,
               dropout_type='standard',
               time_distributed=False,
               selected_frame=-1,
               input_format=None,
               **kwargs):
    """Creates a UNet layer."""
    self._scales = scales
    self._base_filters = base_filters
    self._kernel_size = kernel_size
    self._pool_size = pool_size
    self._rank = rank
    self._block_depth = block_depth
    self._use_deconv = use_deconv
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bias=use_bias
    self._use_add_merge=use_add_merge
    self._use_batch_norm = use_batch_norm
    self._use_sync_bn = use_sync_bn
    self._bn_momentum = bn_momentum
    self._bn_epsilon = bn_epsilon
    self._out_channels = out_channels
    self._out_activation = out_activation
    self._use_global_residual = use_global_residual
    self._use_dropout = use_dropout
    self._dropout_rate = dropout_rate
    self._dropout_type = check_util.validate_enum(
        dropout_type, {'standard', 'spatial'}, 'dropout_type')
    self._time_distributed=time_distributed
    self._selected_frame=selected_frame
    self._use_conv_pool=use_conv_pool
    self._input_format=input_format

    block_config = dict(
        scales=self._scales,
        base_filters=self._base_filters, # To be filled for each scale.
        kernel_size=self._kernel_size,
        pool_size = self._pool_size,
        rank=self._rank,
        block_depth=self._block_depth,
        use_deconv = self._use_deconv,
        activation=self._activation,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_bias=self._use_bias,
        use_add_merge=self._use_add_merge,
        use_conv_pool=self._use_conv_pool,
        use_batch_norm=self._use_batch_norm,
        use_sync_bn=self._use_sync_bn,
        bn_momentum=self._bn_momentum,
        bn_epsilon=self._bn_epsilon,
        out_channels = self._out_channels,
        out_activation = self._out_activation,
        use_global_residual = self._use_global_residual,
        use_dropout=self._use_dropout,
        dropout_rate=self._dropout_rate,
        dropout_type=self._dropout_type,
        time_distributed=self._time_distributed,
        selected_frame=self._selected_frame,
        )

    

    super().__init__(**kwargs)
    self._denoising_block1=TimeUNet(**block_config)
    self._denoising_block2=TimeUNet(**block_config)

  def call(self, inputs, training=None):
    """Runs forward pass on the input tensors."""
    x = inputs
    if self._input_format is not None and self._input_format=='complex':
      x012=self._denoising_block1(tf.concat((x[:,:,:,0:3],x[:,:,:,0+5:3+5]),axis=-1))
      x123=self._denoising_block1(tf.concat((x[:,:,:,1:4],x[:,:,:,1+5:4+5]),axis=-1))
      x234=self._denoising_block1(tf.concat((x[:,:,:,2:5],x[:,:,:,2+5:5+5]),axis=-1))
    else:
      x012=self._denoising_block1(x[:,:,:,0:3])
      x123=self._denoising_block1(x[:,:,:,1:4])
      x234=self._denoising_block1(x[:,:,:,2:5])

    x=tf.concat((x012,x123,x234),axis=-1)
    x=self._denoising_block2(x)
    return x

  def get_config(self):
    """Gets layer configuration."""
    config = {
        'scales': self._scales,
        'base_filters': self._base_filters,
        'kernel_size': self._kernel_size,
        'pool_size': self._pool_size,
        'rank': self._rank,
        'block_depth': self._block_depth,
        'use_deconv': self._use_deconv,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_batch_norm': self._use_batch_norm,
        'use_bias': self._use_bias,
        'use_add_merge':self._use_add_merge,
        'use_conv_pool': self._use_conv_pool,
        'use_sync_bn': self._use_sync_bn,
        'bn_momentum': self._bn_momentum,
        'bn_epsilon': self._bn_epsilon,
        'out_channels': self._out_channels,
        'out_activation': self._out_activation,
        'use_global_residual': self._use_global_residual,
        'use_dropout': self._use_dropout,
        'dropout_rate': self._dropout_rate,
        'dropout_type': self._dropout_type,
        'time_distributed':self._time_distributed,
        'selected_frame':self._selected_frame,
        'input_format': self._input_format,
    }
    base_config = super().get_config()
    return {**base_config, **config}
    
