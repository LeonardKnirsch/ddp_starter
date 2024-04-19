#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray worker node on $(hostname)"
ray start --address $1
sleep infinity
