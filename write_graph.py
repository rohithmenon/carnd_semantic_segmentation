import tensorflow as tf
import main

with tf.Session(graph=tf.Graph()) as sess:
  inp, kp, l3, l4, l7 = main.load_vgg(sess, './vgg')
  out = main.layers(l3, l4, l7, 2)
  writer = tf.summary.FileWriter("output", tf.get_default_graph())
  writer.close()
