import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import numpy as np
import time
import argparse
import logging

tf.reset_default_graph()
data = np.load('../data/BSNIP_left_full.npy')

num_subjects, input_dim, batch_size = data.shape[0], data.shape[1], 32
num_channels_1, num_channels_2, num_channels_3 = 32, 64, 256
latent_dim = 10
# hidden_dim_1, hidden_dim_2, hidden_dim_3 = 128, 64, 32
dropout = 0.5

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)

x = tf.placeholder(tf.float32, [None, input_dim, input_dim, 1])

W_enc_mu = tf.get_variable('W_enc_mu', 
                          shape=[num_channels_3, latent_dim], dtype=tf.float32,
                          initializer=initializer)
b_enc_mu = tf.get_variable('b_enc_mu',shape=[latent_dim], dtype=tf.float32, initializer=initializer)

W_enc_sigma = tf.get_variable('W_enc_sigma',shape=[num_channels_3, latent_dim], dtype=tf.float32, initializer=initializer)
b_enc_sigma = tf.get_variable('b_enc_sigma',shape=[latent_dim], dtype=tf.float32, initializer=initializer)

W_dec = tf.get_variable('W_dec',shape=[latent_dim, num_channels_3], dtype=tf.float32, initializer=initializer)
b_dec = tf.get_variable('b_dec',shape=[num_channels_3], dtype=tf.float32, initializer=initializer)

def e2e_conv(x, out_channels, kernel_w, kernel_h):
#     suffix is the shape of the filter
    conv_dx1 = tf.layers.conv2d(x, out_channels, (kernel_h, 1), activation=tf.nn.relu)
    conv_1xd = tf.layers.conv2d(x, out_channels, (1, kernel_w), activation=tf.nn.relu)
    conv_dx1_dxd = tf.tile([conv_dx1],[1, 1, kernel_h, 1, 1])
    conv_1xd_dxd = tf.tile([conv_1xd],[1, 1, 1, kernel_w, 1])
    sum_dxd = conv_dx1_dxd + conv_1xd_dxd
    
    # reshape for next convolution
    sum_dxd = tf.reshape(sum_dxd, [-1, kernel_w, kernel_h, out_channels])
    return sum_dxd

def e2n_conv(x, out_channels, kernel_w, kernel_h):
    conv_1xd = tf.layers.conv2d(x, out_channels, (1, kernel_w), activation=tf.nn.relu)
    
    return conv_1xd

def e2e_deconv(x, out_channels, kernel_w, kernel_h):
    #Recover r & c
    r = tf.reshape(tf.reduce_mean(x, 1), [batch_size,1, kernel_w,-1])
    print('r', r.shape)
    c = tf.reshape(tf.reduce_mean(x, 2), [batch_size,kernel_w,1,-1])
    print('c', c.shape)
    deconv_dx1 = tf.layers.conv2d_transpose(r, out_channels, (kernel_h, 1), activation=tf.nn.relu)
    deconv_1xd = tf.layers.conv2d_transpose(c, out_channels, (1, kernel_w), activation=tf.nn.relu)

    sum_dxd = (deconv_dx1 + deconv_1xd)/2
    
    # reshape for next convolution
    sum_dxd = tf.reshape(sum_dxd, [-1, kernel_w, kernel_h, out_channels])
    return sum_dxd

def e2n_deconv(x, out_channels, kernel_h, kernel_w):
    conv_1xd = tf.layers.conv2d_transpose(x, out_channels, (kernel_h, 1), activation=tf.nn.relu)
    
    return conv_1xd

def encoder(x):
    print('x', x.shape)
    conv1 = e2e_conv(x, num_channels_1, input_dim, input_dim)
    print('conv1', conv1.shape)
    conv2 = e2e_conv(x, num_channels_1, input_dim, input_dim)
    print('conv2', conv2.shape)
    conv3 = e2n_conv(conv2, num_channels_2, input_dim, input_dim)
    print('conv3', conv3.shape)
    
    #n2g layer
    h = tf.layers.conv2d(conv3, num_channels_3, (input_dim, 1), activation=tf.nn.relu)
    print('h', h.shape)
    h = tf.reshape(h, [-1, num_channels_3])
    mu = tf.matmul(h, W_enc_mu) + b_enc_mu
    log_sigma = tf.matmul(h, W_enc_sigma) + b_enc_sigma
    print('mu', mu.shape)
    print('log_sigma', log_sigma.shape)
    # Latent space
    z = mu+tf.exp(log_sigma/2.)*tf.random_normal(shape=[batch_size,latent_dim])

    return z, mu, log_sigma

def decoder(z):
    h_dec = tf.matmul(z, W_dec) + b_dec
    h_dec = tf.reshape(h_dec, [batch_size, 1,1, num_channels_3])
    print('h_dec', h_dec.shape)
    deconv1 = tf.layers.conv2d_transpose(h_dec, num_channels_2, (1, input_dim), activation=tf.nn.relu)
    print('deconv1', deconv1.shape)
    deconv2 = e2n_deconv(deconv1, num_channels_2, input_dim, input_dim) 
    print('deconv2', deconv2.shape)
#     debug from here
    deconv3 = e2e_deconv(deconv2, num_channels_1, input_dim, input_dim)
    out = e2e_deconv(deconv3, 1, input_dim, input_dim)
    return out

def get_next_batch(data, batch_size):
    idx = np.random.randint(data.shape[0], size=batch_size)
    batch = data[idx,:,:]
    batch = np.reshape(batch, [batch_size, input_dim, input_dim, 1])
    return batch

z, mu, log_sigma = encoder(x)
print('z', z.shape)
out = decoder(z)
print(out.shape)

# Compute KL loss and Reconstruction Loss, find sweet spot of two
reconstruction_loss = tf.reduce_mean(tf.square(x - out),1)
kl_loss = -.5*tf.reduce_sum(1+log_sigma-tf.square(mu)-tf.exp(log_sigma),1)
loss = tf.reduce_mean(reconstruction_loss + 0.0001*kl_loss)

#gradient clipping
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
# gradients = optimizer.compute_gradients(loss)
# capped_gradients = [(tf.clip_by_norm(grad,5.0),var) for grad,var in gradients]
# opt = optimizer.apply_gradients(capped_gradients)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
model = "../models/spring/BrainNetCNN_VAE_0.0001_KL_left_square_matrix.ckpt"

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model)    
    start_time = time.time()
    for epoch in range(10000):
        batch_x = get_next_batch(data, batch_size)
        sess.run([optimizer],feed_dict={x:batch_x})
        if epoch % 50 == 0:
            x_out, loss_out, kl_out, rc_out = sess.run([out, loss, kl_loss, reconstruction_loss], feed_dict={x: batch_x})
            print("Epoch: %i total_loss: %f kl_loss: %f rc_loss: %f" %(epoch, loss_out, np.mean(kl_out), np.mean(rc_out)))
    save_path = saver.save(sess, model)
    print('done saving at',save_path)
