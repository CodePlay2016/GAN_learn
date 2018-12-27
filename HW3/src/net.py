import tensorflow as tf

def discriminator(inpt,gn_stddev, training=True):
    '''
        input_arguments:
            inpt: the inpt image tensor, [batch, width, length, channels]
            gn_stddev: a scalar, the stddev for the gaussian noise added to the input image,
    '''
    with tf.variable_scope('dis',reuse=tf.AUTO_REUSE):
        channels = 32
        inpt = inpt+tf.random_normal(shape=tf.shape(inpt), mean=0.0, stddev=gn_stddev, dtype=tf.float32)
        out = tf.layers.conv2d(inpt, filters=channels, kernel_size=5, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d(out, filters=channels*2, kernel_size=5, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d(out, filters=channels*4, kernel_size=5, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d(out, filters=channels*8, kernel_size=5, strides=2, padding='SAME',kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.flatten(out)
        out = tf.layers.dense(out, 1,kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    return out

def generator(inpt,training=True):
    channels = 32
    with tf.variable_scope('gen',reuse=tf.AUTO_REUSE):
        out = tf.layers.dense(inpt, channels*8*16*16)
        out = tf.nn.relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)
        out  = tf.reshape(out, [-1,16,16,channels*8])

        out = tf.layers.conv2d(out, channels*8, 5, padding='SAME')
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)
        
        out = tf.layers.conv2d_transpose(out, channels*4, 4, 2, padding='SAME')
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d_transpose(out, channels*2, 4, 2, padding='SAME')
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)

        out = tf.layers.conv2d(out, channels, 3, padding='SAME')
        out = tf.nn.leaky_relu(out)
        out = tf.layers.batch_normalization(out,epsilon=1e-5,training=training)
        out = tf.layers.conv2d(out, 3, 3,padding='SAME')
        out = tf.nn.tanh(out)
    return out
    
def loss_fn_d(real_scores, fake_scores):
    '''
    https://www.cnblogs.com/sandy-t/p/7076401.html
    '''
    # d_loss = -tf.reduce_mean(tf.log(real_scores)) -tf.reduce_mean(tf.log(1-fake_scores))
    # d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_scores, labels=tf.ones_like(real_scores)))
    # d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_scores, labels=tf.zeros_like(fake_scores)))
    d_loss_real = tf.reduce_mean(tf.scalar_mul(-1, real_scores))
    d_loss_fake = tf.reduce_mean(fake_scores)
    d_loss=d_loss_real + d_loss_fake
    return d_loss

def loss_fn_g(fake_scores):
    # g_loss = -tf.reduce_mean(tf.log(fake_scores))
    # g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_scores, labels=tf.ones_like(fake_scores)))
    g_loss = tf.reduce_mean(tf.scalar_mul(-1, fake_scores))
    return g_loss

