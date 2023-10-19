import os
import tensorflow as tf
import keras_tuner as kt
from model import layers
import tensorflow_mri as tfmri
import utils.preprocessing_trajectory_gen as preproc_traj
import utils.preprocessing_fastdvdnet_noselect as preproc_fastdvdnet
import utils.preprocessing_rolling_fastdvdnet as preproc_roll
import utils.display_function_fastdvdnet as display_func

class HyperModelFastDVDnet(kt.HyperModel):
  def __init__(self,
            config_model,
            inputs_shape,
            optimizer=tf.keras.optimizers.Adam(),
            loss=tfmri.losses.StructuralSimilarityLoss(rank=2),
            metrics=[tfmri.metrics.PeakSignalToNoiseRatio(rank=2),
                tfmri.metrics.StructuralSimilarity(rank=2)],**kwargs):
     
        super().__init__(**kwargs)
        self.config_model=config_model
        self.inputs_shape=inputs_shape
        self.optimizer=optimizer
        self.loss= loss
        self.metrics=metrics

  def build(self,hp):
    #Define and compile Model
    image_inputs= tf.keras.Input(self.inputs_shape)
    outputs=layers.FastDVDNet(**self.config_model)(image_inputs)
    model=tf.keras.Model(inputs=image_inputs,outputs=outputs)

    model.compile(optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=self.metrics or None,
                    run_eagerly=False)
    return model

  def fit(self, hp, model, datasets,config_preproc,config_traj,**kwargs):
    """Fit the model
    datasets: list of 2 tf.data.Datasets ([0] train and [1] validation)"""
    config_traj_temp=config_traj.copy()

    config_traj_temp['ordering']=hp.Choice('ordering',config_traj['ordering'])
    config_traj_temp['vd_spiral_arms']=hp.Int('vd_spiral_arms',config_traj['vd_spiral_arms'][0],config_traj['vd_spiral_arms'][1])
    config_traj_temp['vd_inner_cutoff']=hp.Float('vdi',config_traj['vd_inner_cutoff'][0],config_traj['vd_inner_cutoff'][1])
    config_traj_temp['pre_vd_outer_cutoff']=hp.Float('pre_vdo',config_traj['pre_vd_outer_cutoff'][0],config_traj['pre_vd_outer_cutoff'][1])
    config_traj_temp['vd_outer_density']=hp.Float('outer_den',config_traj['vd_outer_density'][0],config_traj['vd_outer_density'][1])
    config_traj_temp['vd_type']=hp.Choice('vd_type',['linear','hanning','quadratic'])

    traj_function=preproc_traj.create_traj_fn(**config_traj_temp)
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
    
    if 'saveimages' in kwargs and kwargs['saveimages'] is True:
        display_fn=display_func.display_fn(complex_part='abs',selected_image=-1)
        kwargs['callbacks'].append(tfmri.callbacks.TensorBoardImages(log_dir=os.path.join(exp_dir,'logs'),
            max_images=2,x= dataset_withtransforms[1],display_fn=display_fn))
    # if 'callbacks' not in kwargs:
    #     callbacks=[]
    #     checkpoint_filepath=os.path.join(exp_dir,'ckpt/saved_model')
    #     callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    #                                                     filepath=checkpoint_filepath,
    #                                                     monitor='val_loss',
    #                                                     mode='min',
    #                                                     save_weights_only=False,
    #                                                     save_best_only=True))
    #     callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_dir,'logs')))
    #     kwargs['callbacks'] = callbacks
    print(kwargs)
    history=model.fit(dataset_withtransforms[0],
          epochs=kwargs['epochs'],
          verbose=1,
          callbacks=kwargs['callbacks'],
          validation_data=dataset_withtransforms[1])

    return history

