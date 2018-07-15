import tensorflow as tf
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
import scipy
import time
import sys
from colorama import init
from termcolor import colored

from config import *
from utils import *
#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets('MNIST_data/')

init()

def divide(x, divider):
	return int(math.ceil(float(x) / float(divider)))

# Nvidia's concept - same as tf.nn.lreaky_relu, but supports FP16
def leaky_relu(x, alpha):
	with tf.name_scope('LeakyRelu'):
		alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
		return tf.maximum(x * alpha, x)

def deconv2d(inputs, output_shape, kernel_size=[5, 5], strides=[2, 2], stddev=0.02, name='deconv2d'):
	weights = tf.get_variable(name, [kernel_size[0], kernel_size[1], output_shape[-1], inputs.get_shape()[-1]],
		initializer=tf.random_normal_initializer(stddev=stddev))
	deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape=output_shape, strides=[1, strides[0], strides[1], 1])
	return deconv

def conv2d(inputs, output_dimension, kernel_size=[5, 5], strides=[2, 2], stddev=0.02, name='conv2d'):
	weights = tf.get_variable(name, [kernel_size[0], kernel_size[1], inputs.get_shape()[-1], output_dimension],
		initializer=tf.truncated_normal_initializer(stddev=stddev))
	conv = tf.nn.conv2d(inputs, weights, strides=[1, strides[0], strides[1], 1], padding='SAME')
	return conv

def discriminator(x, training=True, reuse=False):

	with tf.variable_scope('Discriminator', reuse=reuse):

		x = conv2d(x, discriminator_filters, name='d1')
		#x = tf.layers.batch_normalization(x, training=training) # Add or remove?
		x = leaky_relu(x, alpha)

		x = conv2d(x, discriminator_filters*2, name='d2')
		x = tf.layers.batch_normalization(x, training=training)
		x = leaky_relu(x, alpha)

		x = conv2d(x, discriminator_filters*4, name='d3')
		x = tf.layers.batch_normalization(x, training=training)
		x = leaky_relu(x, alpha)

		x = conv2d(x, discriminator_filters*8, name='d4')
		x = tf.layers.batch_normalization(x, training=training)
		x = leaky_relu(x, alpha)

		x = tf.reshape(x, [-1, x.shape[1] * x.shape[2] * discriminator_filters * 8])
		x = tf.layers.dense(x, units=1)

		return x

def generator(batch_size, z_dim, training=True, reuse=False):
	with tf.variable_scope('Generator', reuse=reuse):
		z = tf.random_uniform([batch_size, z_dim], minval=z_dim_range[0], maxval=z_dim_range[1])

		w2, h2 = divide(w, 2), divide(h, 2)
		w4, h4 = divide(w2, 2), divide(h2, 2)
		w8, h8 = divide(w4, 2), divide(h4, 2)
		w16, h16 = divide(w8, 2), divide(h8, 2)
		
		x = tf.layers.dense(z, units=w16 * h16 * generator_filters*8)
		x = tf.reshape(x, [-1, w16, h16, generator_filters*8])
		x = tf.layers.batch_normalization(x, training=training)
		x = leaky_relu(x, alpha)

		x = deconv2d(x, [batch_size, w8, h8, generator_filters*4], name='g1')
		x = tf.layers.batch_normalization(x, training=training)
		x = leaky_relu(x, alpha)

		x = deconv2d(x, [batch_size, w4, h4, generator_filters*2], name='g2')
		x = tf.layers.batch_normalization(x, training=training)
		x = leaky_relu(x, alpha)

		x = deconv2d(x, [batch_size, w2, h2, generator_filters], name='g3')
		x = tf.layers.batch_normalization(x, training=training)
		x = leaky_relu(x, alpha)

		x = deconv2d(x, [batch_size, w, h, 3], name='g4')
		x = tf.nn.tanh(x)

		return x

#'''
#x_input = tf.placeholder(tf.float32, [None, 784]) # For MNIST only, (and other 28x28 images)
x_input = tf.placeholder(tf.float32, [None, w, h, 3])
sample = generator(batch_size, z_dimension)

d_real = discriminator(x_input)
d_fake = discriminator(sample, reuse=True)

if smoothing == False:
	d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_real,
		labels=tf.ones_like(d_real)))

	d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_fake,
		labels=tf.zeros_like(d_fake)))
else:
	d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_real,
		labels=tf.ones_like(d_real) * smooth(one_smoothing[0], one_smoothing[1])))
	d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_fake,
		labels=tf.zeros_like(d_fake) * smooth(zero_smoothing[0], zero_smoothing[1])))

d_loss = d_real_loss + d_fake_loss
#g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=d_real)) # Alternative loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
for i in g_vars:
	print(i.name)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=d_beta1).minimize(d_loss, var_list=d_vars) # How does beta1 affect the output?
	g_optimizer = tf.train.AdamOptimizer(g_lr, beta1=g_beta1).minimize(g_loss, var_list=g_vars) # How does beta1 affect the output?

sampler = generator(1, z_dimension, reuse=True, training=False)

start_time = time.time()
saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	# Load saved model (if exists)
	print(colored('Looking for checkpoint...', 'yellow'))
	current_save_directory = saves_folder + dataset_name + '/'
	saved_data_path = current_save_directory + dataset_name
	if not os.path.exists(current_save_directory):
		if train:
			print(colored('Checkpoint does not exist! Creating new data...', 'yellow'))
		else:
			print(colored('You must train the network before generating images!', 'red'))
			sys.exit(1)
		os.makedirs(current_save_directory)
	else:
		print(colored('Checkpoint directory found! Loading...', 'green'))
		try:
			saver.restore(sess, saved_data_path)
		except:
			if train:
				print(colored('An error occured when loading checkpoint. Creating new data...', 'red'))
			else:
				print(colored('An error occured when loading checkpoint.', 'red'))
				sys.exit(1)
	print(colored('Done.', 'green'))
	# -------------------------------

	if train:
		for i in range(training_steps):

			#bx = mnist.train.next_batch(batch_size)[0]
			bx = np.asarray(get_random_batch(dataset_name, batch_size))
			bx = 2*bx - 1

			sess.run(d_optimizer, feed_dict={x_input: bx})
			sess.run(g_optimizer, feed_dict={x_input: bx})
			sess.run(g_optimizer, feed_dict={x_input: bx})
			
			
			if i % 50 == 0:
				sess.run(g_optimizer, feed_dict={x_input: bx}) # Additional training of generator every 50 iterations
				#sess.run(d_optimizer, feed_dict={x_input: bx}) # Additional training of discriminator every 50 iterations

				generated_sample = sess.run(sampler).reshape([w, h, 3])
				generated_sample = (generated_sample + 1.) / 2.
				plt.imsave('output/%s.png' % i, generated_sample)

				dl = sess.run(d_loss, feed_dict={x_input: bx})
				gl = sess.run(g_loss, feed_dict={x_input: bx})

				duration = time.time() - start_time
				print('time: %s, d_loss: %s, g_loss: %s' % (colored(round(duration, 2), 'yellow'), colored(dl, 'green'), colored(gl, 'red')))
				saver.save(sess, saved_data_path) # WORK IN PROGRESS
	else:
		generated_sample = sess.run(sampler).reshape([w, h, 3])
		generated_sample = (generated_sample + 1.) / 2.
		plt.imsave('output/sample.png' % i, generated_sample)