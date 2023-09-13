import os
import tensorflow as tf
import keras_tuner as kt
from model import layers

import utils.preprocessing_trajectory_gen as preproc_traj
import utils.preprocessing_fastdvdnet_noselect as preproc_fastdvdnet
import utils.preprocessing_rolling_fastdvdnet as preproc_roll
import utils.display_function_fastdvdnet as display_func

class HyperModelFastDVDnet(kt.HyperModel):
  
  def build(self, hp,config_model,inputs_shape):
    #Define and compile Model
    image_inputs= tf.keras.Input(inputs_shape)
    outputs=layers.FastDVDNet(**config_model)(image_inputs)
    model=tf.keras.Model(inputs=image_inputs,outputs=outputs)

    return model

  def fit(self, hp, model, exp_dir, datasets,config_preproc,config_traj,**kwargs):
    """Fit the model
    datasets: list of 2 tf.data.Datasets ([0] train and [1] validation)"""
    traj_function=preproc_traj.create_traj_fn(**config_traj)
    preproc_function=preproc_fastdvdnet.preprocessing_fn(**config_preproc)
    roll_function=preproc_roll.preprocessing_fn()
    
    dataset_withtransforms=[]
    for pp,dataset in enumerate(datasets):
        dataset = dataset.apply(traj_function)
        dataset=dataset.map(preproc_function,num_parallel_calls=1)
        if pp==0:
            dataset=dataset.cache()
        dataset=dataset.map(roll_function,num_parallel_calls=1)
        dataset=dataset.shuffle(buffer_size=8,seed=1)
        if pp>0:
            dataset=dataset.cache()
        dataset=dataset.batch(1,drop_remainder=True)
        dataset=dataset.prefetch(buffer_size=-1)
        dataset_withtransforms.append(dataset)
    if 'epochs' not in kwargs:
        kwargs['epochs'] = 5
    if 'callbacks' not in kwargs:
        callbacks=[]
        checkpoint_filepath=os.path.join(exp_dir,'ckpt/saved_model')
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                                                        filepath=checkpoint_filepath,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_weights_only=False,
                                                        save_best_only=True))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_dir,'logs')))
        kwargs['callbacks'] = callbacks
    history=model.fit(dataset_withtransforms[0],
          epochs=kwargs['epochs'],
          verbose=1,
          callbacks=kwargs['callbacks'],
          validation_data=dataset_withtransforms[1])

    return history

