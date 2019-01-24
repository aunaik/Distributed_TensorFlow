#!/bin/bash
source config.txt
echo "starting servers:"
nohup python tf-server-setup.py $worker1 $worker2 $parameter_server worker 0 >/dev/null 2>&1 &
nohup ssh $second_node "python ~/tensorflow-tutorial/tf-server-setup.py $worker1 $worker2 $parameter_server worker 1 &" >/dev/null 2>&1 &
nohup python tf-server-setup.py $worker1 $worker2 $parameter_server ps 0 >/dev/null 2>&1 &
