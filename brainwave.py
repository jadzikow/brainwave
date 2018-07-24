import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE, SpectralEmbedding

weight_decay = 0.0005
bias_initial_value = 0.01
dropout_rate = 0.20

def split_recording(data, interval_duration, step_size=32):
    n_points = data.values.shape[0]
    sampling_freq = 256 # 256 times per second
    points_per_interval = sampling_freq * interval_duration
    
    n_intervals = int(n_points / points_per_interval)
    
    return np.array([data.values[i * points_per_interval + j : (i + 1) * points_per_interval + j, :] for i in range(n_intervals) for j in range(0, points_per_interval, step_size) if (i + 1) * points_per_interval + j <= data.values.shape[0]])

def load_data(path, class_names, is_directory=False, classes_in_subdirs=False, interval_seconds=2, step_size=32):
    data = []
    labels = []
    
    data_files = []
    if is_directory:
        data_files = [os.path.join(path, file) for file in os.listdir(path) if file.lower().endswith(".csv")]
    else:
        if not path.lower().endswith(".csv"):
            raise Exception("File is not a CSV file!")
        data_files = [path]
        
    for file in data_files:
        batch = pd.read_csv(file)
        split_data = split_recording(batch, interval_seconds, step_size) # Extracting 2 second intervals from the recorded data
        
        data.append(split_data)
        
        class_id = class_names.index("_".join(os.path.basename(file).split("_")[:-1]))
        labels.append(class_id * np.ones(split_data.shape[0]))
    
    data = np.concatenate(data)
    labels = np.concatenate(labels)

    return data, labels
       



def run_FFT(X, sampling_rate=(1.0/256)):
    n = X.shape[1]
    freq = np.fft.fftfreq(n, sampling_rate)
    #freq = freq[1:int(n/2)] # we only care about positive frequencies, starting from n/2 + 1 there are negative freqs and 0 is a sum of the signal
    #freq = freq[1:int(n/4)] # let's take only frequencies up to about 60Hz

    freq = freq[5:int(n/6)] # let's take frequencies in interval [3,45] Hz

    n_desired_freqs = 112
    #freq = freq[:n_desired_freqs]

    print(freq.shape, freq.max())
    print(freq)
    print([i for i in range(len(freq)) if abs(freq[i] - 60) <= 0.5])
    
    fft_transformed = []

    for i in range(X.shape[0]): # THIS CAN BE OPTIMIZED IN NUMPY
        data_point = X[i,:,:]

        # time domain gets transformed into complex numbers where abs(complex_num) represents amplitude of sine
        # and the angle of the complex_num is phase
        transformed = np.fft.fft(data_point, axis=0) / n
        #transformed = transformed[1:int(n/2),:]
        transformed = transformed[5:int(n/6),:] # let's take only frequencies up to about 60Hz
        #transformed = transformed[:n_desired_freqs,:] 
        transformed = abs(transformed) # Let's take the powers of the frequencies, ignore shifts for now

        # let's filter out 60Hz
        #transformed[[i for i in range(len(freq)) if abs(freq[i] - 60) <= 0.5], :] = 0.00001 * np.zeros((1, transformed.shape[1]))
        #transformed[[i for i in range(len(freq)) if abs(freq[i] - 60) <= 0.5], :] *= 0.000001
        #for i in [i for i in range(len(freq)) if abs(freq[i] - 60) <= 0.5]:
        #    for j in range(transformed.shape[1]):
        #        transformed[i, j] = 0
        #print(transformed[[i for i in range(len(freq)) if abs(freq[i] - 60) <= 0.5], :])

        # the below makes 128 features
        #n_band_intervals = 8
        #interval_starts = np.cumsum(4 * np.arange(0,n_band_intervals))
        #interval_ends = np.concatenate([interval_starts[1:], [transformed.shape[0]]])
        
        # the below makes 96 features
        #interval_starts = [0, 4, 12, 22, 44, 70]
        #interval_ends = [4, 12, 22, 44, 70, transformed.shape[0]]
        
        # the below makes 64 features
        #interval_starts = [0, 8, 24, 40]
        #interval_ends = [8, 24, 40, transformed.shape[0]]

        #bands_stats = [np.array([band_interval.min(axis=0), band_interval.max(axis=0), band_interval.mean(axis=0), band_interval.std(axis=0)]) 
        #               for band_interval 
        #               in [transformed[interval_starts[i]:interval_ends[i], :] 
        #                   for i in range(len(interval_starts))]]
        
        # Statistics from equal length band intervals
        #band_interval_width = 8
        #bands_stats = [np.array([band_interval.min(axis=0), band_interval.max(axis=0), band_interval.mean(axis=0), band_interval.std(axis=0)]) 
        #               for band_interval 
        #               in [transformed[i*band_interval_width:min((i+1)*band_interval_width, transformed.shape[0]), :] 
        #                   for i in range(int(np.ceil(transformed.shape[0]/band_interval_width)))]]
        
        #bands_stats = np.concatenate(bands_stats)
        
        #print(bands_stats.shape)
        
        #print("Min:", transformed.min(), "Max:", transformed.max())
        #fft_transformed.append(bands_stats)
        fft_transformed.append(transformed)

    fft_transformed = np.array(fft_transformed)
    print("fft transformed shape", fft_transformed.shape)
    
    return fft_transformed, freq

def standardize_data(X, X_mean=None, X_std=None):
    print("Standardizing data")
    if X_mean is None or X_std is None:
        X_mean = X.mean(axis=(0,1), keepdims=True)
        X_std = X.std(axis=(0,1), keepdims=True)
    print("X max", X.max(), "X min", X.min(), "X mean", X_mean, "X std", X_std)
    return (X - X_mean) / X_std, X_mean, X_std

def normalize_data(X, X_min=None, X_max=None):
    print("Normalizing data")
    if X_min is None or X_max is None:
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
    print(X.shape, X_min.shape, X_max.shape)
    return (X - X_min) / (X_max - X_min), X_min, X_max

def reduce_dimensions(X, dim_reducer=None, n_dimensions=20):
    if dim_reducer is None:
        print("Dimensionality reducer is None! Initializing a new one")
        #dim_reducer = Isomap(n_components=n_dimensions)
        dim_reducer = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, n_components=n_dimensions)
        #dim_reducer = TSNE(n_components=n_dimensions, init='pca', random_state=0)
        #dim_reducer = LocallyLinearEmbedding(n_neighbors=16, n_components=n_dimensions, method='standard')
        dim_reducer.fit(X)
    print("Min:", X.min(), "Max:", X.max())
    X = dim_reducer.transform(X)
    print("Shape after cutting to", n_dimensions, "dimensions:", X.shape)
    print("Min:", X.min(), "Max:", X.max())
    return X, dim_reducer

def preprocess_data(X, Y=None, standardize=False, X_mean=None, X_std=None, reduce_dimensions=False, n_dimensions=20, dim_reducer=None, shuffle=False):
    if shuffle:
        np.random.seed(0)
        random_order = np.random.permutation(X.shape[0])
        X = X[random_order]
        if not (Y is None):
            Y = Y[random_order]
    X = X[:,:,1:-1] # Discarding first and last columns (Timestamps and right AUX)
    
    sampling_rate = 1.0/256 # 256Hz -> default sampling rate on 2016 Muse
    n = X.shape[1]

    # Run FFT
    X, frequencies = run_FFT(X, sampling_rate=sampling_rate)
    
    # Standardization
    if standardize:
        X, X_mean, X_std = standardize_data(X, X_mean, X_std)
    
    X = X.reshape((-1, X.shape[1] * X.shape[2])) # Reshaping because we want to have a 2D matrix
    print("Shape after standardization and reshaping to 2D:", X.shape)
    
    return X, Y, X_mean, X_std, dim_reducer
    
    # Reduce dimensionality
    if reduce_dimensions:
        X, dim_reducer = reduce_dimensionality(X, dim_reducer, n_dimensions)
    
    return X, Y, X_mean, X_std, dim_reducer

def batch_generator(X, Y, batch_size=100):
	idx = 0
	dataset_size = X.shape[0]
	indices = np.random.permutation(dataset_size)
	while idx < dataset_size:
		chunk = slice(idx, idx+batch_size)
		chunk = indices[chunk]
		chunk = sorted(chunk)
		idx = idx + batch_size
		yield X[chunk], Y[chunk]

def parametric_relu(input_, name):
    alpha = tf.get_variable(
        name=name + '_alpha', 
        shape=input_.get_shape()[-1],
        initializer=tf.random_uniform_initializer(minval=0.1, maxval=0.3),
        dtype=tf.float32)
    pos = tf.nn.relu(input_)
    neg = alpha * (input_ - tf.abs(input_)) * 0.5
    return pos + neg

def pool_layer(input_data, pool_size):
    return tf.layers.max_pooling2d(
        inputs=input_data, 
        pool_size=pool_size, 
        strides=2,
        padding="valid")

def fully_connected_layer(name, input_data, units, dropout=True, batch_norm=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layer = tf.layers.dense(
            inputs=input_data,
            units=units,
            #kernel_initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
            #bias_initializer=tf.ones_initializer(),
            bias_initializer=tf.constant_initializer(bias_initial_value),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
            bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))

        #layer = parametric_relu(layer, name + "_prelu_fc") 
        layer = tf.nn.relu(layer)
        if batch_norm:
            layer = tf.layers.batch_normalization(inputs=layer, training=is_training)
        if dropout:
            layer = tf.layers.dropout(inputs=layer, rate=dropout_rate, training=is_training)

        return layer

def brainwave_mlp_model(input_data, n_classes):
    #layer_size = 2048
    layer_size = 4096
    fc1 = fully_connected_layer("fc1", input_data, layer_size, dropout=False, batch_norm=False)
    fc2 = fully_connected_layer("fc2", fc1, layer_size, dropout=False, batch_norm=False)
    fc3 = fully_connected_layer("fc3", fc2, layer_size, dropout=False, batch_norm=False)
    fc4 = fully_connected_layer("fc4", fc3, layer_size, dropout=False, batch_norm=False)
    #fc5 = fully_connected_layer("fc5", fc4, layer_size, dropout=False, batch_norm=False)
    #fc6 = fully_connected_layer("fc6", fc5, layer_size, dropout=False, batch_norm=False)
    
    output = tf.layers.dense(
        name = "classification_logits",
        inputs=fc4, 
        units=n_classes,
        #kernel_initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
        #bias_initializer=tf.ones_initializer(),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
        bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
    
    print("Classificatoin logits name:", output.name)
    return output
    

# With dropout
def train_model(x, y, is_training, neural_network_model, X, Y, test_X, test_Y, n_classes, num_epochs, batch_size, save_path):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
        prediction = neural_network_model(x, n_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        loss += tf.losses.get_regularization_loss()
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        #    optimizer = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())   

            for epoch_cnt in range(num_epochs):
                epoch_loss = 0

                for epoch_x, epoch_y in batch_generator(X, Y, batch_size):                
                    _, c = sess.run([optimizer, loss], feed_dict = {x: epoch_x, y: epoch_y, is_training: True})
                    epoch_loss += c

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                #epoch_validation_accuracy = 0.0
                #for epoch_x, epoch_y in batch_generator(test_data, 'valid', batch_size): 
                #    epoch_validation_accuracy += accuracy.eval({x: epoch_x, y: epoch_y, is_training: False}) * epoch_x.shape[0]
                #epoch_validation_accuracy /= test_data["X_valid"].shape[0]

                predicted_labels = []
                epoch_test_accuracy = 0.0
                for epoch_x, epoch_y in batch_generator(test_X, test_Y, batch_size): 
                    epoch_test_accuracy += accuracy.eval({x: epoch_x, y: epoch_y, is_training: False}) * epoch_x.shape[0]
                    predicted_labels.append(prediction.eval({x: epoch_x, y: epoch_y, is_training: False}))
                epoch_test_accuracy /= test_X.shape[0]
                predicted_labels = np.concatenate(predicted_labels)
                #print(predicted_labels)

                print('Epoch', epoch_cnt + 1, '/', num_epochs, 'Loss:', epoch_loss, "Test:", epoch_test_accuracy)
                #print('Epoch', epoch_cnt + 1, '/', num_epochs, 'Loss:', epoch_loss, "Validation:", epoch_validation_accuracy, "Test:", epoch_test_accuracy)
            
            saver = tf.train.Saver(max_to_keep=10)
            saver.save(sess, save_path)

def save_preprocessing_data(save_path, X_min, X_max):
    np.save(save_path+"_X_min.npy", X_min)
    np.save(save_path+"_X_max.npy", X_max)

def load_preprocessing_data(save_path):
    X_min = np.load(save_path+"_X_min.npy")
    X_max = np.load(save_path+"_X_max.npy")

    return X_min, X_max

def load_model(path):    
    print("Entered load_model")
    sess = tf.Session()   
    saver = tf.train.import_meta_graph(path + ".meta")
    saver.restore(sess, tf.train.latest_checkpoint('saved_models/'))

    return sess
""" 
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    feed_dict ={w1:13.0,w2:17.0}

    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    
    print sess.run(op_to_restore,feed_dict)
 """

#def classify(session, X, Y):
def classify(session, X):
    print("Entered classify")
    graph = tf.get_default_graph()
    print(graph.get_all_collection_keys())

    x = graph.get_tensor_by_name("input_x:0")
    y = graph.get_tensor_by_name("input_y:0")
    is_training = graph.get_tensor_by_name("input_is_training:0")
    classification_op = graph.get_tensor_by_name("classification_logits/BiasAdd:0")
    #predictions = tf.argmax(tf.nn.softmax_cross_entropy_with_logits_v2(logits=classification_op, labels=y), 1)
    predictions = tf.argmax(classification_op, 1)
    correct = tf.equal(predictions, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    
    #feed_dict = {x: X, y: Y, is_training: False}
    feed_dict = {x: X, is_training: False}
    
    #computed_accuracy = session.run([accuracy], feed_dict)
    #return computed_accuracy

    computed_predictions = session.run([predictions], feed_dict)
    return computed_predictions

## SCRIPT CODE
def create_classifier():
    #class_names = ["idle", "right_fist", "left_fist"]
    #class_names = ["look_straight", "look_right", "look_left"]
    class_names = ["idle", "look_right", "look_left", "eyes_closed", "jaw_clench", "smile"]
    n_classes = len(class_names)

    #X, Y = load_data("data/fist_clenching/", class_names, is_directory=True, interval_seconds=2, step_size=32)
    #X, Y = load_data("data/looking/train", class_names, is_directory=True, interval_seconds=2, step_size=32)
    X, Y = load_data("data/6way/train", class_names, is_directory=True, interval_seconds=2, step_size=32)
    X, Y, X_mean, X_std, dim_reducer = preprocess_data(X, Y, shuffle=True)

    #test_X, test_Y = load_data("data", class_names, is_directory=True, interval_seconds=2, step_size=32)
    #test_X, test_Y = load_data("data/fist_clenching/test_data", class_names, is_directory=True, interval_seconds=2, step_size=32)
    #test_X, test_Y = load_data("data/looking/test", class_names, is_directory=True, interval_seconds=2, step_size=32)
    test_X, test_Y = load_data("data/6way/test", class_names, is_directory=True, interval_seconds=2, step_size=32)
    test_X, test_Y, _, _, _ = preprocess_data(test_X, test_Y, shuffle=True)

    X, X_min, X_max = normalize_data(X)
    print(X.shape, X_min.shape, X_max.shape)
    test_X, _, _ = normalize_data(test_X, X_min, X_max)
    print(test_X.shape)

    Y_onehot = np.zeros((len(Y), n_classes))
    Y_onehot[np.arange(len(Y)), Y.astype(int)] = 1
    Y = Y_onehot
    test_Y_onehot = np.zeros((len(test_Y), n_classes))
    test_Y_onehot[np.arange(len(test_Y)), test_Y.astype(int)] = 1
    test_Y = test_Y_onehot

    n_features = X.shape[1]
    x = tf.placeholder('float', [None, n_features], name="input_x")
    y = tf.placeholder('float', name="input_y")
    is_training = tf.placeholder('bool', shape=[], name="input_is_training")
    print(x.name, y.name, is_training.name)

    saved_model_path = "saved_models/brainwave-MLP-model"
    saved_preprocessing_data_path = "saved_models/brainwave-MLP-model"

    train_model(x, y, is_training, brainwave_mlp_model, X, Y, test_X, test_Y, n_classes=n_classes, num_epochs=4, batch_size=1000, save_path=saved_model_path)
    save_preprocessing_data(saved_preprocessing_data_path, X_min, X_max)

    loaded_session = load_model(saved_model_path)
    X_min, X_max = load_preprocessing_data(saved_preprocessing_data_path)

    return loaded_session, X_min, X_max

    #test_X, test_Y = load_data("data/looking/test", class_names, is_directory=True, interval_seconds=2, step_size=32)
    test_X, test_Y = load_data("data/6way/test", class_names, is_directory=True, interval_seconds=2, step_size=32)
    test_X, test_Y, _, _, _ = preprocess_data(test_X, test_Y, shuffle=True)
    test_X, _, _ = normalize_data(test_X, X_min, X_max)

    test_Y_onehot = np.zeros((len(test_Y), n_classes))
    test_Y_onehot[np.arange(len(test_Y)), test_Y.astype(int)] = 1
    test_Y = test_Y_onehot

    #classify(loaded_session, test_X, test_Y)

    return loaded_session, X_min, X_max

def preprocess_and_classify(session, test_X, X_min, X_max):
    test_X, _, _, _, _ = preprocess_data(test_X)
    test_X, _, _ = normalize_data(test_X, X_min, X_max)

    return classify(session, test_X)

#classifier, X_min, X_max = create_classifier()
#test_X, test_Y = load_data("data/6way/test", ["idle", "look_right", "look_left", "eyes_closed", "jaw_clench", "smile"], is_directory=True, interval_seconds=2, step_size=32)
#preprocess_and_classify(classifier, test_X, X_min, X_max)
