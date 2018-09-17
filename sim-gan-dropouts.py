
import os
import sys
import matplotlib.pyplot as plt
import random
from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import glob
from dlutils import plot_image_batch_w_labels
from scipy import misc

from utils.image_history_buffer import ImageHistoryBuffer


import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.insert(0, '../')
from scipy import misc
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import LocallyConnected2D
from keras.layers.convolutional import Conv2D
from keras.layers import *
from keras import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard, Callback
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
import cv2
from common import util
from sklearn.model_selection import train_test_split
import random
import glob
import tensorflow as tf
K.set_image_dim_ordering('th')


#
# directories
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')

#
# image dimensions
#

img_width = 640
img_height = 480
img_channels = 1

#
# training params
#

nb_steps = 10000 #low for now
batch_size = 2
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 500


  def gen_samples(directory,shuffle = True, directories=[],real=True):
        samples = []
        dirs = os.listdir(directory) if len(directories)==0 else filter(lambda x: x in directories,os.listdir(directory))
        for i in dirs:
            path = os.path.join(directory, i)+"/"
            if os.access(path, os.R_OK):
                if real:
                    samples.extend(sorted(glob.glob(path+"*_depth.png")))
                else:
                    samples.extend(sorted(glob.glob(path+"*_truth.png")))

        if shuffle:
            random.shuffle(samples)
        return samples  

    def generate_data_custom_depth(samples,img_height=480,img_width=640,batch_size=4,func=[]):
    i = 0
    while True:
        stack1 = np.zeros((batch_size,img_height,img_width,1))
        stack2 = np.zeros((batch_size,img_height,img_width,1))
        j=0
        while j<batch_size:
            try: 
                rgb = samples[i][0]
                depth = samples[i][2]
                rgb_img = grab_frame1(rgb,normalize)
                depth_img = grab_frame1(depth,hot_vectorize)
                rgb_img,depth_img = augment(rgb_img,depth_img,func,bias_to_real = .5)
                stack1[j] = np.reshape(rgb_img,(img_height,img_width,1))
                stack2[j] = np.reshape(depth_img,(img_height,img_width,1))
                j+=1
                i= (i+1)%len(samples)
            except Exception:
                i=(i+1)%len(samples)
        yield (stack1,stack2)

    def hot_vectorize(x,value = 0):
        zero_mask = x==value
        non_zero_mask = x!=value
        x[zero_mask] = 1
        x[non_zero_mask] = 0
        return x.astype(float)

    def stack_frames(frames,img_height,img_width,channels):
        stack  = np.zeros((1,img_height,img_width,channels))
        index = 0
        for i in frames:
            num_chan = 1 if len(np.shape(i)) ==2 else np.shape(i)[2]
            stack[0,:,:,index:index+num_chan] = np.reshape(i,(img_height,img_width,num_chan))
            index += num_chan
        return stack

    def grab_frame(files,i,path,func=None):
        img = misc.imread(path+files[i])
        if func:
            return func(img)
        return img

    def grab_frame1(path,func=None):
        img = misc.imread(path)
        if func:
            return func(img)
        return img

    def normalize(x):
        x=x.astype(float)/3500.
        x*=2
        x-=1
        return x

    def normalize_depth(x):
        return x.astype(float)/3000.

    def convert_rgb_normal(img):
        return (img/255.*2)-1

    def bounding_box(img,size = 100):
        h,w = np.shape(img)
        non_zeros = np.nonzero(img)
        x_min = np.min(non_zeros[0])
        x_max = np.max(non_zeros[0])
        y_min = np.min(non_zeros[1])
        y_max = np.max(non_zeros[1])
        out = (x_min,x_min+size,y_min,y_min+size)            
        rgb = sorted(glob.glob(path+"r-*"))
        #if size else (x_min,x_max,y_min,y_max)#minuce or plus coordinates
        if x_min< 0 or x_min+size > h or y_min<0 or y_min+size>w:
            return None
        return out

    def crop(x,x1 = 100,x2 = 500,y1 = 50, y2 = 450):
        x = x[y1:y2,x1:x2]
        return x
  
    def gen_samples(directory,key,shuffle = True):
            samples = []
            dirs = os.listdir(directory)
            for i in dirs:
                path = os.path.join(directory, i)+"/"
                if os.access(path, os.R_OK):
                    depth = sorted(glob.glob(path+"*"+key))
                    samples.extend(depth)
            if shuffle:
                random.shuffle(samples)
            return samples

    def generate_data_custom_depth(samples,img_height=img_height,img_width=img_width,batch_size=4,func=[]):
        i = 0
        while True:
            stack1 = np.zeros((batch_size,img_height,img_width,1))
            j=0
            while j<batch_size:
                try: 
                    depth = samples[i]
                    depth_img = grab_frame1(depth,normalize)
                    # depth_img = crop(depth_img)
                    stack1[j] = np.reshape(depth_img,(img_height,img_width,1))
                    j+=1
                    i= (i+1)%len(samples)
                except Exception:
                    i=(i+1)%len(samples)
            yield stack1 

    # samples = gen_samples("/media/drc/DATA/chris_labelfusion/RGBDCNN/",key="_truth.png")
    # synthetic_generator = generate_data_custom_depth(samples,img_height=img_height,img_width=img_width,batch_size= batch_size)

    # samples = gen_samples("/media/drc/DATA/chris_labelfusion/RGBDCNN/",key ="_depth.png")
    # real_generator = generate_data_custom_depth(samples,img_height=img_height,img_width=img_width,batch_size= batch_size)
 

    def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch

def refiner_network(input_image_tensor):
   

    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = layers.Convolution2D(nb_features, nb_kernel_cols, padding='same',data_format = "channels_last")(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(nb_features, nb_kernel_rows, padding='same',data_format = "channels_last")(y)

        y = layers.add([input_features, y])
        return layers.Activation('relu')(y)

    x = layers.Convolution2D(64, (3, 3), padding='same', activation='relu',data_format = "channels_last")(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for _ in range(5):
        x = resnet_block(x)

    return layers.Convolution2D(img_channels, (1, 1), padding='same', activation='tanh',data_format = "channels_last")(x)

def encoder_decoder_network(input_image_tensor):
    conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(input_image_tensor)
    conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv1)

    conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(pool1)
    conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv2)

    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(pool2)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv3)

    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(pool3)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv5)
    #drop5 = Dropout(0.5)(conv5)

    up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(UpSampling2D(size = (2,2),data_format='channels_last')(conv5))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(merge7)
    conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv7)

    up8 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(UpSampling2D(size = (2,2),data_format='channels_last')(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(merge8)
    conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv8)

    up9 = Conv2D(4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(UpSampling2D(size = (2,2),data_format='channels_last')(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(merge9)
    conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid',data_format='channels_last')(conv9)
    return conv10

def discriminator_network(input_image_tensor):
    x = layers.Convolution2D(96, (3, 3), padding='same', strides=(2, 2), activation='relu',data_format = "channels_last")(input_image_tensor)
    x = layers.Convolution2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu',data_format = "channels_last")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), padding='same', strides=(1, 1),data_format = "channels_last")(x)
    x = layers.Convolution2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu',data_format = "channels_last")(x)
    x = layers.Convolution2D(32, (1, 1), padding='same', strides=(1, 1), activation='relu',data_format = "channels_last")(x)
    x = layers.Convolution2D(2, (1, 1), padding='same', strides=(1, 1), activation='relu',data_format = "channels_last")(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)

def discriminator_network_custom(input_image_tensor):

    x = layers.Convolution2D(96, (7, 7), strides=4, padding='same', activation='relu')(input_image_tensor)
    x = layers.Convolution2D(64, (5, 5), strides=2, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3),strides=2, padding='same')(x)
    x = layers.Convolution2D(32, (3, 3),  strides = 2, padding='same', activation='relu')(x)
    x = layers.Convolution2D(32, (1, 1), strides = 2, padding='same', activation='relu')(x)
    x = layers.Convolution2D(2, (1, 1), strides = 2,padding='same', activation='relu')(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)

def adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path=None, discriminator_model_path=None):
 
    synthetic_image_tensor = layers.Input(shape=(img_height, img_width,img_channels))
    refined_image_tensor = encoder_decoder_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width,img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    #
    # define models
    #

    refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
    discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output,
                                       name='discriminator')

    # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
    refiner_model_output = refiner_model(synthetic_image_tensor)
    print np.shape(refiner_model_output)
    combined_output = discriminator_model(refiner_model_output)
    combined_model = models.Model(input=synthetic_image_tensor, output=[refiner_model_output, combined_output],
                                  name='combined')

    discriminator_model_output_shape = discriminator_model.output_shape

    print(refiner_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    #
    # define custom l1 loss function for the refiner
    #

    def self_regularization_loss(y_true, y_pred):
        delta = 0.0001  # FIXME: need to figure out an appropriate value for this
        return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

    #
    # define custom local adversarial loss (softmax for each image section) for the discriminator
    # the adversarial loss function is the sum of the cross-entropy losses over the local patches
    #

    def local_adversarial_loss(y_true, y_pred):
        # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
        # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return tf.reduce_mean(loss)

    def binary_crossentropy(target, output, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
          Arguments:
              target: A tensor with the same shape as `output`.
              output: A tensor.
              from_logits: Whether `output` is expected to be a logits tensor.
                  By default, we consider that `output`
                  encodes a probability distribution.
          Returns:
              A tensor.
          """
        #target  = crop1(target)
       # output = crop1(output)
        #print np.shape(target), np.shape(output)

          # Note: nn.softmax_cross_entropy_with_logits
          # expects logits, Keras expects probabilities.
        _EPSILON = 10e-8
        if not from_logits:
            # transform back to logits
            epsilon_ =  tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype)
            output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
            output = math_ops.log(output / (1 - output))
        return nn.weighted_cross_entropy_with_logits(target, output, .15, name=None)

    #
    # compile models
    #

    sgd = optimizers.SGD(lr=0.001)

    
    refiner.compile(optimizer = optimizers.Adam(lr = 1e-4), loss =  binary_crossentropy, metrics = ['accuracy'])
    discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss)
    discriminator_model.trainable = False
    combined_model.compile(optimizer=sgd, loss=[binary_crossentropy, local_adversarial_loss])

    
    #data generators
    
    #normalize to 1??????

    # datagen = image.ImageDataGenerator(
    #     #preprocessing_function=applications.xception.preprocess_input,
    #     rescale = 1/3500.,
    #     data_format="channels_last")

    # flow_from_directory_params = {'target_size': (img_height, img_width),
    #                               'color_mode': 'grayscale',
    #                               'class_mode': None,
    #                               'batch_size': batch_size}

    # synthetic_generator = datagen.flow_from_directory(
    #     directory=synthesis_eyes_dir,
    #     **flow_from_directory_params
    # )

    # real_generator = datagen.flow_from_directory(
    #     directory=mpii_gaze_dir,
    #     **flow_from_directory_params
    # )
    samples = gen_samples1("/media/drc/DATA/chris_labelfusion/RGBDCNN/")#directories=['2017-06-16-30']
    train = generate_data_custom_depth(samples,img_height=480,img_width=640)
  

    print np.shape(get_image_batch(synthetic_generator))
    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
    print(np.shape(y_real))
    print y_real
    y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
    assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)
    if not refiner_model_path:
        print('pre-training the refiner network...')
        gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        for i in range(500):
            synthetic_image_batch = get_image_batch(synthetic_generator)
            gen_loss = np.add(refiner_model.train_on_batch(synthetic_image_batch, synthetic_image_batch), gen_loss)
            # log every `log_interval` steps
            print i
            if not i % log_interval:
                figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(i)
                print('Saving batch of refined images during pre-training at step: {}.'.format(i))

                synthetic_image_batch = get_image_batch(synthetic_generator)
                plot_image_batch_w_labels.plot_batch(
                    np.concatenate((synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch))),
                    os.path.join(cache_dir, figure_name),
                    label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

                print('Refiner model self regularization loss: {}.'.format(gen_loss / log_interval))
                gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        refiner_model.save(os.path.join(cache_dir, 'refiner_model_pre_trained.h5'))
    else:
        refiner_model.load_weights(refiner_model_path)

    print("pretrained refiner network")
    
    if not discriminator_model_path:
        print('pre-training the discriminator network...')
        disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

        for _ in range(200):
            real_image_batch = get_image_batch(real_generator)
            disc_loss = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss)
            synthetic_image_batch = get_image_batch(synthetic_generator)
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
            disc_loss = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined), disc_loss)

        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))

        # hard-coded for now
        print('Discriminator model loss: {}.'.format(disc_loss / (100 * 2)))
    else:
        discriminator_model.load_weights(discriminator_model_path)

    # TODO: what is an appropriate size for the image history buffer?
    image_history_buffer = ImageHistoryBuffer((0, img_height, img_width, img_channels), batch_size * 100, batch_size)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the refiner
        for _ in range(k_g * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)


            combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,[synthetic_image_batch, y_real]), combined_loss)

        for _ in range(k_d):
            # sample a mini-batch of synthetic and real images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            real_image_batch = get_image_batch(real_generator)
            
            # refine the synthetic images w/ the current refiner
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

            # use a history of refined images
            half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
            image_history_buffer.add_to_image_history_buffer(refined_image_batch)

            if len(half_batch_from_image_history):
                refined_image_batch[:batch_size // 2] = half_batch_from_image_history

            disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
            disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                       disc_loss_refined)

        if not i % log_interval:
            # plot batch of refined images w/ current refiner
            figure_name = 'refined_image_batch_step_{}.png'.format(i)
            print('Saving batch of refined images at adversarial step: {}.'.format(i))

            synthetic_image_batch = get_image_batch(synthetic_generator)
            plot_image_batch_w_labels.plot_batch(
                np.concatenate((synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch))),
                os.path.join(cache_dir, figure_name),
                label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

            #plt.imshow(np.reshape(refiner_model.predict_on_batch(synthetic_image_batch)[0],(224,224)))
            #plt.show()
            # log loss summary
            print('Refiner model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * k_d * 2)))
            print('Discriminator model loss refined: {}.'.format(disc_loss_refined / (log_interval * k_d * 2)))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_step_{}.h5')
            refiner_model.save(model_checkpoint_base_name.format('refiner', i))
            discriminator_model.save(model_checkpoint_base_name.format('discriminator', i))


def main(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path):
    adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path)


if __name__ == '__main__':
    # TODO: if pre-trained models are passed in, we don't take the steps they've been trained for into account

    main("/media/drc/DATA/chris_labelfusion/logs_test/test_gan_sim","/media/drc/DATA/chris_labelfusion/logs_test/test_gan",None,None)