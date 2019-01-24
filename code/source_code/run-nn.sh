#!/bin/bash

source config.txt

layers=50,25,10
learning_rate=0.01
num_epoch=10

python MNIST_NN.py $worker1 $worker2 $parameter_server $layers $learning_rate $num_epoch 
