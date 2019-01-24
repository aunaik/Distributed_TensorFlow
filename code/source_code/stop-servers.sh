#!/bin/bash
source config.txt
kill -9 $(lsof -t -i:$port1)
ssh $second_node "source ~/tensorflow-tutorial/config.txt && kill -9 \$(/usr/sbin/lsof -t -i:$port2)"
kill -9 $(lsof -t -i:$port3)

