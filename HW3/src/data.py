import tensorflow as tf
from PIL import Image
import os, pdb
HEIGHT = 64
WIDTH = 64
CHANNEL = 3

def _bytes_feature(value):
    """
    generate byte feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_tfrecord(source_path):
    """
    generate tf_record for basic GAN model
    https://blog.csdn.net/sinat_34474705/article/details/78966064
    """
    writer = tf.python_io.TFRecordWriter(os.path.join(source_path , "train.tfrecords"))
    img_dir = os.path.join(source_path, 'faces')
    for imageName in os.listdir(img_dir):
        image = Image.open(os.path.join(img_dir,imageName))
        image = image.resize((64,64),Image.BILINEAR)
        pdb.set_trace()
        image_raw = image.tobytes() # convert image to binary format
        example = tf.train.Example(features = tf.train.Features(feature = {
            "image_raw": _bytes_feature(image_raw),
            }))
        writer.write(example.SerializeToString())
    writer.close()

def readRecord(recordName):
    """
    read TFRecord data (images).

    Arguments:
    recordName -- the TFRecord file to be read.
    return: data saved in recordName 
    """
    filenameQueue = tf.train.string_input_producer([recordName])
    reader = tf.TFRecordReader()
    _, serializedExample = reader.read(filenameQueue)
    features = tf.parse_single_example(serializedExample, features={
        "image_raw": tf.FixedLenFeature([], tf.string)
    })

    image = features["image_raw"]
    image = tf.image.decode_jpeg(image, tf.uint8)
    image = preprocess(image)
    return image

def preprocess(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    print('image_record shape before process', image.get_shape().as_list())
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    print('image_record shape after process', image.get_shape().as_list())
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image

def get_batch_image(data, batch_size, shuffle=True):
    '''
    input:
        data: list of datas, such as [image, label]
    '''
    if shuffle: return tf.train.shuffle_batch(data, batch_size=batch_size, capacity=2*batch_size, min_after_dequeue=batch_size)
    return tf.train.batch(data, batch_size, capacity=2*batch_size)

def get_batch_noise(dimension, batch_size):
    noise = tf.random_normal([batch_size,dimension],mean=0.0,stddev=1.0)
    return noise

if __name__ == '__main__':
    img_path = '../data/'
    generate_tfrecord(img_path)
    image = readRecord(os.path.join(img_path,'train.tfrecords'))
    batch = get_batch_image([image], 10)
    sess = tf.Session()

    print('')
