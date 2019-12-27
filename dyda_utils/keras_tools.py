import os
import numpy as np
import scipy as sp
from sklearn import cluster
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Activation, merge
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import np_utils
from dyda_utils import image
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda_utils import data


def predict_with_uncertainty(f, x, no_classes, n_iter=100):
    """ Get prediction result with uncertainty using dropout
        See event_classifier.py for sample how to use it
    """

    result = np.zeros((n_iter,) + (x.shape[0], no_classes))
    y = np.expand_dims(x, axis=0)

    for i in range(n_iter):
        result[i, :, :] = f((y, 1))[0]

    prediction = result.mean(axis=0).mean(axis=0)
    uncertainty = result.std(axis=0).mean()
    return prediction, uncertainty


def train_data_generator(data_dir, img_width, img_height, batch_size=32,
                         class_mode="categorical", train_mode=True):
    """Return ImageDataGenerator for training data

    @param data_dir: path of the images
    @param img_width: image width
    @param img_height: image height

    Arguments:
    class_mode -- class mode for y labels, can be "categorical",
                  "binary", "sparse" (default: categorical)
    batch_size -- Batch size (default: 32)
    classes   -- A pre-defined list of class index (default: None)
    convert_Y -- True to use np_utils.to_categorical to convert Y
                 (default: True)
    sort      -- True to sort the images (default: False)

    """

    if train_mode:
        datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    else:
        datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode)

    return generator


def val_data_generator(val_data_dir, img_width, img_height,
                       class_mode="categorical", batch_size=32):
    """return ImageDataGenerator for validation data"""

    return train_data_generator(
        val_data_dir, img_width, img_height, train_mode=False,
        class_mode=class_mode, batch_size=batch_size)


def prepare_cifar10_data(nb_classes=10):
    """ Get Cifar10 data """

    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test


def prepare_mnist_data(rows=28, cols=28, nb_classes=10):
    """ Get MNIST data """

    from keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
    X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test


def visualize_model(model, to_file='model.png'):
    '''Visualize model (work with Keras 1.0)'''

    if type(model) == Sequential or type(model) == Model:
        from keras.utils.visualize_util import plot
        plot(model, to_file=to_file)

    elif type(model) == Graph:
        import pydot
        graph = pydot.Dot(graph_type='digraph')
        config = model.get_config()
        for input_node in config['input_config']:
            graph.add_node(pydot.Node(input_node['name']))

        for layer_config in [config['node_config'],
                             config['output_config']]:
            for node in layer_config:
                graph.add_node(pydot.Node(node['name']))
                if node['inputs']:
                    for e in node['inputs']:
                        graph.add_edge(pydot.Edge(e, node['name']))
                else:
                    graph.add_edge(pydot.Edge(node['input'], node['name']))

        graph.write_png(to_file)


def extract_hypercolumn(model, la_idx, instance):
    ''' Extract HyperColumn of pixels (Theano Only)

    @param model: input DP model
    @param la_idx: indexes of the layers to be extract
    @param instamce: image instance used to extract the hypercolumns

    '''
    import theano
    layers = [model.layers[li].get_output(train=False) for li in la_idx]
    get_feature = theano.function([model.layers[0].input], layers,
                                  allow_input_downcast=False)
    feature_maps = get_feature(instance)
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)


def is_dense(layer):
    '''Check if the layer is dense (fully connected)
       Return layer name if it is a dense layer, None otherwise'''

    layer_name = layer.get_config()['name']
    ltype = layer_name.split('_')[0]
    if ltype == 'dense':
        return layer_name
    return None


def is_convolutional(layer):
    '''Check if the layer is convolutional
       Return layer name if it is a dense layer, None otherwise'''

    layer_name = layer.get_config()['name']
    ltype = layer_name.split('_')[0]
    if ltype.find('convolution') > -1:
        return layer_name
    return None


def cluster_hc(hc, n_jobs=1):
    ''' Use KMeans to cluster hypercolumns'''

    ori_size = hc.shape[1]
    new_size = ori_size*ori_size
    m = hc.transpose(1, 2, 0).reshape(new_size, -1)
    kmeans = cluster.KMeans(n_clusters=2, max_iter=300, n_jobs=n_jobs,
                            precompute_distances=True)
    cluster_labels = kmeans .fit_predict(m)
    imcluster = np.zeros((ori_size, ori_size))
    imcluster = imcluster.reshape((new_size,))
    imcluster = cluster_labels
    return imcluster.reshape(ori_size, ori_size)


def get_class_from_path(opath, keyword=''):
    """Get object class from the file path

    @param opath: path of the object
    @param keyword: keyword of the classes to search

    """
    _dirname = os.path.dirname(opath)
    while len(_dirname) > 1:
        base = os.path.basename(_dirname)
        if keyword == '':
            return base
        elif keyword is not None and base.find(keyword) > -1:
            return base
        _dirname = os.path.dirname(_dirname)
    return None


def prepare_data(img_loc, width, height, convert_Y=True,
                 rc=False, scale=True, classes=None, sort=False):
    """ Read images as dp inputs

    @param img_loc: path of the images or a list of image paths
    @param width: number rows used to resize the images
    @param height: number columns used to resize the images

    Arguments:
    rc        -- True to random crop the images as four (default: False)
    scale     -- True to divide input images by 255 (default: True)
    classes   -- A pre-defined list of class index (default: None)
    convert_Y -- True to use np_utils.to_categorical to convert Y
                 (default: True)
    sort      -- True to sort the images (default: False)

    """

    print('[dyda_utils] width = %i, height = %i' % (width, height))
    if type(img_loc) is list:
        imgs = img_loc
    else:
        imgs = image.find_images(dir_path=img_loc)
    X = []
    Y = []
    F = []
    create_new_cls = False
    if classes is None:
        create_new_cls = True
        classes = []
    counter = 0

    if rc:
        print('[dyda_utils] Applying random crop to the image')
    if sort:
        imgs = sorted(imgs)
    for fimg in imgs:
        if counter % 1000 == 0:
            print('[dyda_utils] Reading images: %i' % counter)
        _cls_ix = get_class_from_path(fimg)
        if _cls_ix not in classes and create_new_cls:
            classes.append(_cls_ix)

        if rc:
            _img_original = image.read_and_random_crop(
                fimg, size=(height, width)).values()
        else:
            _img_original = [image.read_img(fimg, size=(height, width))]

        if _img_original[0] is None:
            continue
        for img in _img_original:
            X.append(img)
            Y.append(classes.index(_cls_ix))
            F.append(fimg)
        counter += 1
        fname = str(counter) + '.jpg'

    X = np.array(X).astype('float32')

    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], X.shape[-1], height, width)
        input_shape = (X.shape[-1], height, width)
    else:
        X = X.reshape(X.shape[0], height, width, X.shape[-1])
        input_shape = (height, width, X.shape[-1])

    if scale:
        X /= 255
    if convert_Y:
        Y = np_utils.to_categorical(np.array(Y), len(classes))

    return np.array(X), np.array(Y), classes, F, input_shape


def prepare_data_test(img_loc, width, height, convert_Y=True,
                      scale=True, classes=None, y_as_str=True):
    """ Read images as dp inputs

    @param img_loc: path of the images or a list of image paths
    @param width: number rows used to resize the images
    @param height: number columns used to resize the images

    Arguments:
    y_as_str  -- True to return Y as a list of class strings
                 This overwrites convert_Y as False. (default: True)
    convert_Y -- True to use np_utils.to_categorical to convert Y
                 (default: True)

    """
    if y_as_str:
        X, Y, classes, F, input_shape = prepare_data(
            img_loc, width, height, sort=True,
            scale=scale, classes=classes, convert_Y=False)
        _Y = [classes[_y] for _y in Y]
        return X, _Y, classes, F, input_shape
    X, Y, classes, F, input_shape = prepare_data(
        img_loc, width, height, scale=scale,
        classes=classes, convert_Y=convert_Y, sort=True)
    return X, Y, classes, F, input_shape


def prepare_data_inference(img_path, width, height):
    """ Read images as dp inputs

    @param img_path: path of the images or a list of image paths
    @param width: number rows used to resize the images
    @param height: number columns used to resize the images

    """
    X, Y, classes, F, input_shape = prepare_data(
        img_path, width, height, sort=True,
        scale=True, classes=None, convert_Y=False)
    return X, F


def split_samples(data, target, test_size):
    """Split samples

    @param data: Input full data array (multi-dimensional np array)
    @param target: Input full target array (1D np array)

    """
    from sklearn import cross_validation

    train_d, test_d, train_t, test_t = \
        cross_validation.train_test_split(data, target, test_size=test_size)
    return train_d, test_d, train_t, test_t


def prepare_data_train(img_loc, width, height, sort=False,
                       test_size=None, rc=False,
                       scale=True, classes=None):
    """ Read images as dp inputs

    @param img_loc: path of the images or a list of image paths
    @param width: number rows used to resize the images
    @param height: number columns used to resize the images

    Arguments:

    sort      -- True to sort the images (default: False)
    test_size -- size of the testing sample (default: 0.33)

    """

    X, Y, classes, F, input_shape = prepare_data(
        img_loc, width, height, rc=rc,
        scale=scale, classes=classes, sort=sort)

    X_train, X_test, Y_train, Y_test = split_samples(X, Y, test_size)
    print('[dyda_utils] X_train shape: ', X_train.shape)
    print('[dyda_utils] Y_train shape: ', Y_train.shape)
    print('[dyda_utils] %i train samples' % X_train.shape[0])
    print('[dyda_utils] %i test samples' % X_test.shape[0])

    return X_train, X_test, Y_train, Y_test, classes, input_shape


def output_model_info(model_path, model_assets={}, note="",
                      model_type="classification", outpath="./model.json"):
    """ Output model info based on dyda_utils spec https://goo.gl/So46Jw

    @param model_path: File path of the model

    Arguments:

    model_assets -- Model assets, details see dyda_utils spec
    note         -- Note for the model
    model_type   -- Type of the model based on softmax outputs
                    detection, classification
    outpath      -- File path of the output json

    """

    model_info = {"framework": "keras", "model_type": model_type}

    if not tools.check_exist(model_path):
        print('[dyda_utils] ERRPR: %s does not exist' % model_path)
        return

    model_info["model_path"] = model_path
    model_info["model_sha256sum"] = tools.get_sha256(model_path)
    model_info["model_assets"] = model_assets
    model_info["note"] = note

    data.write_json(model_info, fname=outpath)


def _output_pred(input_path):
    """ Output prediction result based on dyda_utils spec https://goo.gl/So46Jw

    @param input_path: File path of the input

    """

    return data._output_pred(input_path)


def output_pred_classification(input_path, conf, label, labinfo={}):
    """ Output classification result based on spec https://goo.gl/So46Jw

    @param input_path: File path of the input
    @param conf: Confidence score
    @param label: Label of the result

    Arguments:

    labinfo -- Additional results

    """

    lab_tools.output_pred_classification(input_path, conf, label,
                                         labinfo=labinfo)
