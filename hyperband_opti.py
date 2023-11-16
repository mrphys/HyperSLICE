#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday May 10 10:11:42 2023
Example Code for optimizing spiral trajectory jointly with Image Deep Artifact Suppression network (FastDVDnet) for interactive MRI

Methods details in : 
HyperSLICE: HyperBand optimised Spiral for Low-latency Interactive Cardiac Examination, (2023)

Trained from flower image dataset.
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

@author: Dr. Olivier Jaubert
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import tensorflow as tf
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('running on CPU')
import tensorflow_mri as tfmri
import random
import matplotlib.pyplot as plt
import datetime
import pathlib
import keras_tuner as kt

import model.hypermodel as hypermodel
# Local imports (works if you are in project folder)
import model.layers as layers
import utils.preprocessing_natural_images as preproc_filename_2_kspace
import utils.preprocessing_trajectory_gen as preproc_traj
import utils.preprocessing_fastdvdnet_noselect as preproc_fastdvdnet
import utils.preprocessing_rolling_fastdvdnet as preproc_roll
import utils.display_function_fastdvdnet as display_func


#Set seed for all packages
seed_value=1
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
("")


# In[2]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')


# In[3]:


#Configuration
DEBUG=False
learning_rate=0.0001

config_traj=preproc_traj.config_optimized_traj()
config_preproc=preproc_fastdvdnet.config_base_preproc()
config_natural_images={'base_resolution':config_preproc['base_resolution'],'phases':config_preproc['phases'],'num_coils':10,'addmotion':1}

config={'experiment_path': 'HyperBand_folder',
        'experiment_name': 'Test_FastDVDnet',
        'split' : [0.7,0.15,0.15], #train, val, test
        'split_mode': 'noshuffle', #noshuffle or random
        'learning_rate': learning_rate,
        'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1),
        'epochs':200,
        'loss': tfmri.losses.StructuralSimilarityLoss(rank=2),
        'metrics':[tfmri.metrics.PeakSignalToNoiseRatio(rank=2),
                   tfmri.metrics.StructuralSimilarity(rank=2)]}

imshape=(config_preproc['phases'],config_preproc['base_resolution'],config_preproc['base_resolution'])

config_model={'scales': 3,
      'block_depth': 2,
      'base_filters': 32,
      'kernel_size': 3,
      'use_deconv': 'PixelShuffle',
      'rank': 2,
      'activation': tf.keras.activations.relu,
      'out_channels': 1,
      'kernel_initializer': tf.keras.initializers.HeUniform(seed=1),
      'time_distributed': False}


# In[4]:


# Read files and split data 
train_files=[]
val_files=[]
test_files=[]
sorted_files=[x for x in sorted(list(map(str,data_dir.glob('roses/*'))))]
if DEBUG:
      n=20
else:
      n=len(sorted_files); 
ntrain=int(config['split'][0]*n); nval=int(config['split'][1]*n); ntest=int(np.ceil(config['split'][2]*n))

train_files=sorted_files[:ntrain]
val_files=sorted_files[ntrain:ntrain+nval]
test_files=sorted_files[ntrain+nval:ntrain+nval+ntest]

# Shuffle files.
random.shuffle(train_files)
random.shuffle(val_files)
random.shuffle(test_files)

print('Total/Train/Val/Test:',len(train_files)+len(val_files)+len(test_files),
      '/',len(train_files),'/',len(val_files),'/',len(test_files),'leftovers:',n-ntrain-nval-ntest)


# In[5]:


#Define Preprocessing run once to get input shapes
preproc_natural_image=preproc_filename_2_kspace.preprocessing_fn(**config_natural_images)
traj_function=preproc_traj.create_traj_fn(**config_traj)
preproc_function=preproc_fastdvdnet.preprocessing_fn(**config_preproc)
roll_function=preproc_roll.preprocessing_fn()
# Run Preprocessing once on case [0]
kspace=preproc_natural_image(train_files[1])
ds=tf.data.Dataset.from_tensors(kspace)
image=traj_function(ds)
for element in image:
  inputs_temp,gt_temp=preproc_function(element)
  inputs,gt=roll_function(inputs_temp,gt_temp)
plt.figure(figsize=(12,3))
plt.imshow(np.abs(np.concatenate((inputs[:,:,0],inputs[:,:,1],inputs[:,:,2],inputs[:,:,3],inputs[:,:,4],gt[:,:,0]),axis=1)),cmap='gray')
plt.axis('off')


# In[6]:


#Creating Tensorflow dataset
# Create datasets.
datasets=[train_files,val_files,test_files]
prepdatasets=[]
for pp,dataset in enumerate(datasets):
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(list(map(str, dataset)), dtype=tf.string))
    dataset=dataset.map(preproc_natural_image,num_parallel_calls=1)
    prepdatasets.append(dataset)


# In[7]:


#Define hyperparameters
config_traj['ordering']=['tiny','linear']
config_traj['vd_spiral_arms']=[12,24]
config_traj['vd_inner_cutoff']=[0.1,0.4]
config_traj['pre_vd_outer_cutoff']=[0.01,1.0]
config_traj['vd_outer_density']=[0.01,0.35]
config_traj['vd_type']=['linear','hanning','quadratic']


# In[8]:


#HyperModel FastDVDnet
config_traj_hp=config_traj
hpmodel=hypermodel.HyperModelFastDVDnet(config_model,
                    inputs.shape,
                    optimizer=config['optimizer'],
                    loss=config['loss'],
                    metrics=config['metrics'],name='HM', tunable=True)
#model=hpmodel.build( _ )
# hpmodel.fit( _, model, exp_dir, prepdatasets,config_preproc,config_traj)

#Define Hyperparameters outside for HyperBand
hp = kt.HyperParameters()
hp.Choice('ordering',config_traj['ordering'])
hp.Int('vd_spiral_arms',config_traj['vd_spiral_arms'][0],config_traj['vd_spiral_arms'][1])
hp.Float('vdi',config_traj['vd_inner_cutoff'][0],config_traj['vd_inner_cutoff'][1])
hp.Float('pre_vdo',config_traj['pre_vd_outer_cutoff'][0],config_traj['pre_vd_outer_cutoff'][1])
hp.Float('outer_den',config_traj['vd_outer_density'][0],config_traj['vd_outer_density'][1])
hp.Choice('vd_type',['linear','hanning','quadratic'])

if DEBUG:
    # tuner = kt.RandomSearch(
    #     objective=kt.Objective('val_loss', "min"),
    #     max_trials=5,
    #     hypermodel=hpmodel,
    #     directory="results",
    #     project_name="custom_training",
    #     overwrite=True,
    # )
    max_epochs=5
    tuner = kt.Hyperband(
        hypermodel=hpmodel,
        objective=kt.Objective('val_loss', "min"),
        max_epochs=max_epochs,
        factor=3,
        hyperband_iterations=1,
        directory="results2",
        project_name="debug_hyperband",
        overwrite=True,
        seed=0,
        hyperparameters=hp,
    )
else:
    max_epochs=150
    tuner = kt.Hyperband(
        objective=kt.Objective('val_loss', "min"),
        factor=5,
        hyperband_iterations=1,
        max_epochs=max_epochs,
        hypermodel=hpmodel,
        directory="results2",
        project_name="hyperband",
        overwrite=True,
        hyperparameters=hp,
        seed=0,
    )


# In[9]:


#Launch HyperParameter optimisation
callbacks=[]
tuner.search(prepdatasets,config_preproc,config_traj,callbacks=callbacks)


# In[10]:


#Retrain Best model
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
best_trial.summary()
best_model = tuner.load_model(best_trial)

path = config['experiment_path']
exp_name = os.path.splitext(os.path.basename(config['experiment_name']))[0]
exp_name += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = os.path.join(path, exp_name)

print(exp_dir)
callbacks=[]
checkpoint_filepath=os.path.join(exp_dir,'ckpt/saved_model')
callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        save_best_only=True))
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(exp_dir,'logs')))

print(best_hps.values)
model = tuner.hypermodel.build(best_hps)
history = hpmodel.fit(best_hps,model, prepdatasets,config_preproc,config_traj,callbacks=callbacks,epochs=np.max(50,max_epochs))


# In[11]:


# %reload_ext tensorboard

# %tensorboard --port=6008 --logdir /home/oj20/UCLjob/Project5/GitHub_repo/HyperSLICE/HyperBand_folder/Test_FastDVDnet_20230914_153342


# In[12]:


fig = plt.figure(figsize=(16,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
leg = plt.legend()



# In[21]:


config_best_traj=preproc_traj.config_optimized_traj()
best_traj_config=best_hps.values
config_best_traj['ordering']=best_traj_config['ordering']
config_best_traj[ 'vd_spiral_arms']=best_traj_config[ 'vd_spiral_arms']
config_best_traj['vd_inner_cutoff']=best_traj_config['vdi']
config_best_traj['pre_vd_outer_cutoff']=best_traj_config['pre_vdo']
config_best_traj['vd_outer_density']=best_traj_config['outer_den']
config_best_traj['vd_type']=best_traj_config['vd_type']

traj_function=preproc_traj.create_traj_fn(**config_best_traj)
preproc_function=preproc_fastdvdnet.preprocessing_fn(**config_preproc)


# In[23]:


#Save Configuration
import json
global_config={**config,**config_best_traj,**config_preproc,**config_model}
for key in global_config.keys():
    global_config[key]=str(global_config[key])
filename = os.path.join(exp_dir,'config.json')
with open(filename, 'w') as f:
    f.write(json.dumps(global_config))


# In[24]:


#Inference
#Preproc series 1
print(best_hps.values)
model = tuner.hypermodel.build(best_hps)
model.load_weights(checkpoint_filepath)
kspace=preproc_natural_image(test_files[1])
ds=tf.data.Dataset.from_tensors(kspace)
image=traj_function(ds)
for element in image:
  inputs_temp,gt_temp=preproc_function(element)
#Preproc series 2
kspace2=preproc_natural_image(test_files[2])
ds2=tf.data.Dataset.from_tensors(kspace2)
image2=traj_function(ds2)
for element in image2:
  inputs_temp2,gt_temp2=preproc_function(element)

#Run model on buffered 5 image in a series
inputs=np.concatenate((inputs_temp,inputs_temp2),axis=2)
gts=np.concatenate((gt_temp,gt_temp2),axis=2)
buffer=[]
output=[]
bestoutput=[]
for pp in range(inputs.shape[-1]):
  buffer.append(inputs[:,:,pp])
  if pp>3:
    model_input=np.expand_dims(np.stack(buffer,axis=-1),axis=0)
    output.append(model(model_input))
    bestoutput.append(best_model(model_input))
    buffer=buffer[1:]

output=np.concatenate(output,axis=-1)
bestoutput=np.concatenate(bestoutput,axis=-1)
plot_image=np.concatenate((inputs[:,:,4:],gts[:,:,4:],output[0,...],bestoutput[0,...]),axis=0)

plt.figure(figsize=(15,12))
start_frame=7
plt.imshow(np.abs(np.concatenate((plot_image[:,:,start_frame],plot_image[:,:,start_frame+1],plot_image[:,:,start_frame+2],plot_image[:,:,start_frame+3],plot_image[:,:,start_frame+4],plot_image[:,:,start_frame+5]),axis=1)),cmap='gray')
plt.axis('off')
plt.title('Six consecutive frames with change: Top Input, Mid GT, Bottom Recon')
savefilename=os.path.join(exp_dir,'fig_orientation_change')
plt.savefig(savefilename)
# In[25]:


#From Left to Right: Input, Ground Truth, Reconstructed.
savepath=os.path.join(exp_dir,'video_orientation_change')
display_func.plotVid(np.transpose(plot_image,axes=[1,0,2]),interval=55,vmin=0,vmax=1,savepath=savepath)

