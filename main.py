from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

data_dir = '/tmp/tensorflow/mnist/input_data'

def main(_):
    # Import data
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    # Create the model
    images = tf.placeholder(tf.float32, [None, 784], name="images")
    weights = tf.Variable(tf.zeros([784, 10]), name="weights")
    #for each pixel there are 10 different weights for each digit
    bias = tf.Variable(tf.zeros([10]), name="bias")
    evidence = tf.matmul(images, weights) + bias

    # Define loss and optimizer
    digits = tf.placeholder(tf.float32, [None, 10], name="digits")

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=digits, logits=evidence))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for i in range(1000):
        batch_images, batch_labels = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={images: batch_images, digits: batch_labels})
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(evidence, 1), tf.argmax(digits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={images: mnist.test.images,
                                        digits: mnist.test.labels}))

if __name__ == '__main__':
    tf.app.run(main=main)
