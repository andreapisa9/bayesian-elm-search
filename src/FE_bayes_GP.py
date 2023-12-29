import train
import keras_model

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from copy import copy
import tensorflow_datasets as tfds

WORKING_DS = 'MNIST'

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def create_model(blocks, block_code, residuals):

    # Initializer seed
    initializer = tf.keras.initializers.HeNormal(seed=69)

    # Define model according to hyperparameters
    inputs = tf.keras.layers.Input([28,28,1])
    x = inputs

    act_outs = []
    for b in range(int(blocks)):

        # Decoding definition of block structure #TODO: debug!
        code = str(int(block_code))[b * 4 : (b + 1) * 4]
        filters = int(code[0]) % 3
        kernel_size = int(code[1]) % 3
        padding = int(code[2]) % 2
        regularization = int(code[3]) % 2

        if filters == 0:
            filters = 16
        elif filters == 1:
            filters = 32
        elif filters == 2:
            filters = 64

        if kernel_size == 0:
            kernel_size = 3
        elif kernel_size == 1:
            kernel_size = 5
        elif kernel_size == 2:
            kernel_size = 7

        if padding or blocks > 4:
            padding = 'same'
        else:
            padding = 'valid'

        if regularization:
            regularization = tf.keras.regularizers.l2(1e-4)
        else:
            regularization = None

        # Adding block to model
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=regularization
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn{b}')(x)

        
        if residuals >= .5 and (b % 3 == 0) and (b >= 3):
            if x.shape == act_outs[b-2].shape:
                x = (tf.keras.layers.Add()([x, act_outs[b-2]]))

        x = tf.keras.layers.Activation('relu', name=f'a{b}')(x)
        act_outs.append(x)

    if int(blocks):
        pool_size = int(np.amin(x.shape[1:3]))
        x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)
    y = tf.keras.layers.Flatten(name="extract")(x)

    # Compile model
    model = tf.keras.Model(inputs=inputs, outputs=y)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics='accuracy',
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None
    )

    return model

def black_box_function(blocks, block_code, residuals):
    global REF_EXTRACTION

    model = create_model(blocks, block_code, residuals)

    # Return a very bad score if output shape is different from reference (similarity is not computable)
    if model.output_shape[1] != REF_EXTRACTION.shape[1]:
        return -1e6

    # Extract features of train set and compare to standard extraction
    prediction = model.predict(train_data, verbose=0)
    return -mse(REF_EXTRACTION, prediction)

def mse(a: np.ndarray, b: np.ndarray):
    return (np.square(a - b)).mean()

if __name__ == '__main__':
    # Load dataset
    if WORKING_DS == 'CIFAR-10':
        train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names\
            = train.load_cifar_10_data('cifar-10-batches-py') # CAUTION: labels are one-hot encoded. Proceed consequently
    elif WORKING_DS == 'CIFAR-100':
        train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names\
            = train.load_cifar_100_data('cifar-100-python') # CAUTION: labels are one-hot encoded. Proceed consequently
    elif WORKING_DS == 'MNIST':
      (train_data, test_data), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
      )
      train_data = train_data.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
      )
      train_data = train_data.cache()
      train_data = train_data.shuffle(ds_info.splits['train'].num_examples)
      #train_data = tf.reshape(train_data, [28,28,1])
      train_data = train_data.batch(128)
      train_data = train_data.prefetch(tf.data.AUTOTUNE)
      test_data = test_data.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
      )
      #test_data = tf.reshape(test_data, [28,28,1])
      test_data = test_data.batch(128)
      test_data = test_data.cache()
      test_data = test_data.prefetch(tf.data.AUTOTUNE)
    
    if 'CIFAR' in WORKING_DS and not(os.path.exists(f'./{WORKING_DS}/perf_samples_idxs.npy')):
        get_perf_samples_idxs.main()
    if 'CIFAR' in WORKING_DS:
      _idxs = np.load(os.path.join(os.getcwd(), f'{WORKING_DS}/perf_samples_idxs.npy'))
      test_data_bm = test_data[_idxs]
      test_labels_bm = test_labels[_idxs]
      test_filenames_bm = test_filenames[_idxs]

    # Load reference extractor
    feature_extractor_model = keras_model.resnet_v1_eembc() # ds=WORKING_DS
    #feature_extractor_model.load_weights(os.path.join(os.getcwd(), f'{WORKING_DS}/trained_models/pretrainedResnet.h5'))
    feature_extractor_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None
    )
    FE_layer = "extract"

    feature_extractor_model.fit(
      train_data,
      epochs=6,
      validation_data=test_data
    )

    feature_extractor_model.save_weights(os.path.join(os.getcwd(), f'{WORKING_DS}/trained_models/pretrainedResnet.h5'))

    reference = tf.keras.Model(inputs=feature_extractor_model.inputs,
                        outputs=feature_extractor_model.get_layer(FE_layer).output)
    
    REF_EXTRACTION = reference.predict(train_data)

    # Initialize logger object to save progress
    logger = JSONLogger(path=f'./{WORKING_DS}/logs.log', reset=False)
    
    pbounds = {'blocks': [0.0, 10.0], 'block_code': [1e40, (1e41 - 1)], 'residuals': [0.0, 1.0]}

    # Launch Bayesian search
    optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=69)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # If previous search is logged, optimizer is loaded with prior experience
    if os.path.exists(os.path.join(os.getcwd(), f'{WORKING_DS}/logs.log.json')):
        pass # load_logs(optimizer, logs=[f'./{WORKING_DS}/logs.log.json'])
    
    optimizer.maximize(init_points=0, n_iter=2e4, acquisition_function=UtilityFunction(kind='ucb', kappa=1.96))

    print(f"Best result: {optimizer.max['params']}; f(x) = {optimizer.max['target']}")

    # Save best model
    best_fe = create_model(
        optimizer.max['params']['blocks'],
        optimizer.max['params']['block_code'],
        optimizer.max['params']['residuals']
    )

    best_fe.save(f"./{WORKING_DS}/trained_models/ELM_FE.h5")