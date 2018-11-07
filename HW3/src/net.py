import tensorflow as tf

def discriminator(inpt,gn_stddev, training=True):
    '''
        input_arguments:
            inpt: the inpt image tensor, [batch, width, length, channels]
            gn_stddev: a scalar, the stddev for the gaussian noise added to the input image,
    '''
    with tf.variable_scope('dis',reuse=tf.AUTO_REUSE):
        fs = 16
        batch_size, height, width, _ = inpt.shape.as_list()
        
        inpt = inpt+tf.random_normal(shape=tf.shape(inpt), mean=0.0, stddev=gn_stddev, dtype=tf.float32)
        out = tf.layers.conv2d(inpt, filters=fs, kernel_size=4, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d(out, filters=fs*2, kernel_size=4, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d(out, filters=fs*4, kernel_size=4, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d(out, filters=fs*8, kernel_size=4, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)
        print(out.get_shape().as_list())

        out = tf.layers.flatten(out)
        out = tf.layers.dense(out, 1, activation=tf.nn.sigmoid,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    return out

def generator(inpt,training=True):
    fs = 8
    with tf.variable_scope('gen',reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(inpt, fs*8*4*4, activation=tf.nn.relu)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)
        out  = tf.reshape(out, [-1,4,4,fs*8])
        
        out = tf.layers.conv2d_transpose(out, fs*4, 4, 2, padding='SAME')
        out = tf.nn.relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d_transpose(out, fs*2, 4, 2, padding='SAME')
        out = tf.nn.relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d_transpose(out, fs, 4, 2, padding='SAME')
        out = tf.nn.relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d_transpose(out, 3, 4, 2,padding='SAME')
        out = tf.nn.tanh(out)
    return out
    
def loss_fn_d(real_scores, fake_scores):
    '''
    https://www.cnblogs.com/sandy-t/p/7076401.html
    '''
    d_loss = -tf.reduce_mean(tf.log(real_scores)) -tf.reduce_mean(tf.log(1-fake_scores))
    return d_loss

def loss_fn_g(fake_scores):
    g_loss = -tf.reduce_mean(tf.log(fake_scores))
    return g_loss

