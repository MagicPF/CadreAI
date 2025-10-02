#!/bin/bash

# Repeat running the Python script 100 times
for i in {1..100}
do
    echo "Run #$i"
    python experiment_druggen_MT2.py
done