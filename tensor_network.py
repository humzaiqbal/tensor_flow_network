import tensorflow as tf 
import numpy as np 
from sklearn.preprocessing import normalize
"""
This class acts as a wrapper for the tensorflow neural network 

"""

class tensor_Network:
	def __init__(self, learning_rate = 0.01, num_iterations=2000, DROPOUT = 0.5):
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations
		self.DROPOUT = DROPOUT

	def __one_hot_encode(self, labels):
		encoded_labels = []
		unique_labels = np.unique(labels)
		num_classes = len(unique_labels) 
		for label in labels:
			label_vector = [0] * num_classes
			location = np.argwhere(unique_labels == label)[0][0]
			label_vector[location] = 1
			encoded_labels.append(label)
		return encoded_labels

	def __normalize_features(self, data):
		return normalize(data, axis=0)

	def __get_batch(dataset, labels, batch_size):
		batch_data = []
		batch_labels = []
		zipped = zip(dataset, labels)
		packed_samples = random.sample(zipped, batch_size)
		for elem in packed_samples:
			batch_data.append(elem[0])
			batch_labels.append(elem[1])
		return batch_data, batch_labels

	def train(datset, labels):
		num_features = len(dataset[0])
		num_classes = len(np.unique(labels))
		x = tf.placeholder("float", shape=[None, num_features])
		y_ = tf.placeholder("float", shape=num_classes)
		W = tf.Variable(tf.zeros([num_features, num_classes]))
		b = tf.Variable(tf.zeroes([num_classes]))
		y = tf.nn.softmax(tf.matmul(x, W) + b)
		cross_entropy = -tf.reduce_sum(y_*tf.log(y))
		train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		predict = tf.argmax(y, 1)
		init = tf.initialize_all_variables()
		keep_prob = tf.placeholder("float")
		while tf.Session() as sess:
			sess.run(init)

			for i in range(self.num_iterations):
				batch_data, batch_labels = self.__get_batch(dataset, labels)
				sess.run(train_step, feed_dict={x: batch_data, y_:batch_labels}, keep_prob: self.DROPOUT)
			





