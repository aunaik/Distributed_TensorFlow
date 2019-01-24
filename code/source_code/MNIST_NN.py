import math
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import timeit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Random_mini_batches
def random_mini_batches(X, Y, mini_batch_size = 32, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data
    Y -- true "label" vector
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)            # To make "random" minibatches the same every time
    m = X.shape[1]                  # number of training examples
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


#cost computation
def compute_cost(Z, Y):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    Z -- vector containing z, output of the last linear unit
    Y -- vector of true label 
    
    Returns:
    cost
    """
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


# one hot matrix
def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    C = tf.constant(C, name="C" )
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot


# Creating placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector
    n_y -- scalar, number of classes
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    """

    X = tf.placeholder(tf.float32, [n_x, None], name="Placeholder_1")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Placeholder_2")

    return X, Y


# Initializing parmeters
def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    tf.set_random_seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dims[l],layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l],1], initializer = tf.zeros_initializer())

    return parameters


# Forward propogation
def forward_propagation(X, parameters):
    """
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    Z_l -- last pre-activation value
    """
    A = {}
    L = len(parameters) // 2                     # number of layers in the neural network
    A['A1'] = tf.nn.relu(tf.add(tf.matmul(parameters["W1"], X),  parameters["b1"]))
    for l in range(2, L):
        A['A'+str(l)] = tf.nn.relu(tf.add(tf.matmul(parameters["W"+str(l)], A['A'+str(l-1)]),  parameters["b"+str(l)]))
    Z_l = tf.add(tf.matmul(parameters["W"+str(L)], A['A'+str(L-1)]),  parameters["b"+str(L)])

    return Z_l


# Main Function
def model(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32,
          print_cost = True):
    """
    Implements a multi-layer tensorflow neural network
    
    Arguments:
    X_train -- training set, of shape (input size = 784, number of training examples = 60000)
    Y_train -- test set, of shape (output size = 10, number of training examples = 60000)
    X_test -- training set, of shape (input size = 784, number of training examples = 10000)
    Y_test -- test set, of shape (output size = 10, number of test examples = 10000)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tf.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    #with tf.Graph().as_default():
    #	tf.reset_default_graph()	

    #with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z_l = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z_l, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    start = timeit.default_timer()

    # Start the session to compute the tensorflow graph
    with tf.Session("grpc://"+ worker1, config = tf.ConfigProto(allow_soft_placement= True)) as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                #print (minibatch_cost)
                #exit()
                epoch_cost += minibatch_cost / num_minibatches
            # Print the cost every epoch
            #if print_cost == True and epoch % 100 == 0:
            f.write("Cost after epoch %i: %f\n" % (epoch, epoch_cost) )
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        stop = timeit.default_timer()

        # plot the cost
        #plt.plot(np.squeeze(costs))
        #plt.ylabel('cost')
        #plt.xlabel('iterations (per tens)')
        #plt.title("Learning rate =" + str(learning_rate))
        #plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        #print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z_l), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        f.write("\n")
        f.write("Train Accuracy: {}%\n".format(accuracy.eval({X: X_train, Y: Y_train})*100))
        f.write("Test Accuracy: {}%\n".format(accuracy.eval({X: X_test, Y: Y_test})*100))
        f.write("Calculation time: {} secs\n".format(np.round(stop-start,2)))
        f.close()

    return parameters


worker1 = sys.argv[1]
worker2 = sys.argv[2]
ps = sys.argv[3]

#cluster_spec = tf.train.ClusterSpec({"worker" : [worker1,worker2], "ps" : [ps]})


mnist = keras.datasets.mnist
(X_train_orig, Y_train_orig) , (X_test_orig, Y_test_orig) = mnist.load_data()
classes = 10
# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = one_hot_matrix(Y_train_orig, 10)
Y_test = one_hot_matrix(Y_test_orig, 10)


layer_dims = np.array(map(int, sys.argv[4].split(',')))
learning_rate = float(sys.argv[5])
num_epochs = int(sys.argv[6])

# Adding input layer to the dimensions
layer_dims = np.concatenate((np.array([X_train.shape[0]]),layer_dims))
#print (layer_dims)
#print (learning_rate)
#print (num_epochs)
#exit()
f= open("../../output_data/TF/test_output.txt","w+")
f.write("Layers Dimensions = {}\n".format(layer_dims))
f.write("Learning rate = {}\n".format(learning_rate))
f.write("Number of epochs = {}\n\n".format(num_epochs))

# Call to model
parameteres = model(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate ,num_epochs, minibatch_size = 32, print_cost = True)
