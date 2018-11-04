import tensorflow as tf

def _parse_function(example_proto):
    features = {"image_raw": tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["image_raw"]

r_path = '/Users/hufangquan/code/GAN_learn/HW3/data/train.tfrecords'

data = tf.data.TFRecordDataset(r_path)
data = data.map(_parse_function)
data = data.repeat()
data = data.batch(32)
iterator = data.make_initializable_iterator()
image = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer)
i = sess.run(image)
print('')

