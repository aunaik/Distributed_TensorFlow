# TensorFlow

## Installation Instructions

If TensorFlow is not installed for the user on the allotted node, run the following command
```
pip install --upgrade --user tensorflow
```

If Keras is not installed for the user on the allotted node, run the following command

```
pip install --upgrade --user keras
```

## Different files in source code

#### config.txt
Specify worker and parameter server details with node and port information.

#### tf-server-setup.py
Contains code for assigning job type and task index to the workers and parameter servers by creating a 'ClusterSpec' object and passing 'ClusterSpec' object to 'Server' object to identify the local task with job name and task index

#### start-server.py 
We stat server for each task by running this file. Running this file activates the nodes we have configured in config.txt

#### stop-server.py 
This file kills the task we have started by using start-server.py after execution of our neural network code

#### run-nn.sh 
this file contains the hyperparameter we are tuning for the experiment. For trying different Neural network architecture we have to make changes in this code and run it

## Execution Instructions

### Steps:
Start the servers using the following code.
```
./start-server.py
```

Once the node is up, start the neural network training by using the following command
```
./run-nn.sh
```
   Note: For changing the Neural network architecture as well as number of epochs make changes to this file.

Once the execution of neural network is complete and the output file is saved in the output folder. Kill the tasks by using the following command
```
./stop-server.py
```
Note: Sample outputs for all the test-cases run are present in the output folder
 
