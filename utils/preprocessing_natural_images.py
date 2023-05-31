import tensorflow as tf
import tensorflow_mri as tfmri
import tensorflow_addons as tfa
import math
rg = tf.random.Generator.from_seed(1, alg='philox')

def preprocessing_fn(base_resolution=128,
                      phases=20,num_coils=10,addmotion=1):
  """Returns a preprocessing function for training."""
  
  def _preprocessing_fn(filename):
    """From jpg to formatted kspace of image.

    Takes natural images and maps to magnitude only images.

    Args:
      filename: RGB jpg filename [x,y,3]
        
    Returns:
      Ground truth multi coil kspace.
    """
    with tf.device('/gpu:0'):
      image_init=tf.io.read_file(filename)
      image_init= tf.cast(tf.io.decode_png(image_init,channels=3),tf.float32)
      #image_init=tf.cast(image.imread(str(filename)),tf.float32)
      image_init=tf.math.sqrt(tf.reduce_sum(image_init**2,axis=-1))

      #Create multiple frames
      image_series=tf.tile(tf.expand_dims(image_init,axis=0),[phases,1,1])
      
      if addmotion > 0 :
        #Add translational motion between frames
        image_series=tf.expand_dims(image_series,axis=-1)
        image_series=_geometric_augmentation(image_series,phases,motion_proba=1,motion_ampli=0.5,maxrot=0.0)
        image_series=image_series[...,0]
      image_series=tfmri.resize_with_crop_or_pad(image_series,shape=[base_resolution,base_resolution])
      image_series=tf.cast(image_series,tf.complex64)

      # Simulate coil sensitivity maps
      smaps = []
      angle=0
      for coil in range(num_coils):
          angle =angle+2*math.pi/num_coils
          temp = gauss_kernel(2,base_resolution, sigma = base_resolution/4)
          transform=(base_resolution/3*tf.cos(angle),base_resolution/3*tf.sin(angle))
          smaps.append(tfa.image.translate(temp,transform,'bilinear' ))
      smaps=tf.stack(smaps,axis=0)
      sos_smaps=tf.abs(tf.sqrt(tf.reduce_sum(tf.complex(smaps[:,:,:,0],smaps[:,:,:,1])*tf.complex(smaps[:,:,:,0],-smaps[:,:,:,1]),axis=0)))
      smaps=tf.complex(smaps[:,:,:,0],smaps[:,:,:,1])
      #sos_smaps=tf.cast(sos_smaps,tf.complex64)
      smaps = smaps / tf.cast(tf.reduce_max(sos_smaps),tf.complex64)
      
      #Create Coil Images
      coil_images=tf.expand_dims(smaps,axis=1)*tf.expand_dims(image_series,axis=0)
      sos_image=tf.abs(tf.sqrt(tf.reduce_sum(coil_images*tf.math.conj(coil_images),axis=0)))
      coil_images=coil_images/tf.cast(tf.reduce_max(sos_image),tf.complex64)
      #Create ground truth kspace and sos ground truth image
      #sos_image=sos_image/tf.reduce_max(sos_image)
      kspace=tfmri.signal.fft(coil_images, axes=[-2, -1], norm='ortho', shift=True)
      kspace=tf.cast(kspace,tf.complex64)
      kspace=tf.transpose(kspace,[1,0,2,3])
      
      # Dictionary with cartesian multi-coil kspace data in 'kspace' field
      #Format [slice, phases, coils, x, y]
      output=dict()
      output['kspace']=tf.expand_dims(kspace,axis=0)
    return output

  return _preprocessing_fn

def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel
@tf.function
def _geometric_augmentation(image,phases,motion_proba=0.5,motion_ampli=0.8,maxrot=45.0):
  transform=tf.zeros((tf.shape(image)[0],2))
  image_size=tf.shape(image)
  if rg.uniform(shape=())<motion_proba:
    displacement=[]
    for idxdisp in range(phases):
        if idxdisp%5==0:
            transform=rg.uniform(shape=(2,1), minval=-motion_ampli, maxval=motion_ampli)
        else:
            transform=transform
        if idxdisp==0:
            displacement.append(tf.cast([[0], [0]],tf.float32))
        else:
            displacement.append(transform+displacement[idxdisp-1])

    transform=tf.stack(displacement)
    transform=tf.squeeze(transform)
    #print(image.shape,transform.shape,transform.dtype)
    image=tfa.image.translate(image,transform,'bilinear' )

  if maxrot>0:
    #ROTATION
    rotationangle=rg.uniform(shape=(),minval=-maxrot,maxval=maxrot)
    
    image=tfa.image.rotate(image, angles=rotationangle,interpolation='bilinear')
    cpx_size=tf.shape(image)
    image=image[:,cpx_size[1]//2-image_size[1]//2:(cpx_size[1]//2-image_size[1]//2+image_size[1]),
            cpx_size[2]//2-image_size[2]//2:(cpx_size[2]//2-image_size[2]//2+image_size[2]),...]
  return image

