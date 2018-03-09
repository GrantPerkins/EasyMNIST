# Have some notes:
https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginners

Every image is 28x28 → 784px

mnist.train.images is [55000, 784]
	55000 images, 784px/image
  
mnist.train.labels is  [55000, 10]
	55000 images, 10 labels possible
	[[0,0,0,0,0,1,0,0,0,0],...]
 
softmax → algo that gives probabilities of a set amount of options → good for 10 MNIST labels

softmax → add evidence of being in certain class, convert evidence to probabilities

Steps for softmax regression:
1. Find positive and negative weights for each pixel belonging to a given class (0→ 9)
2. Add bias for given class
3. Plug evidence tallies into softmax function
4. Evidence = Weights * input + bias
