#!/bin/bash
round=3
# Repeat running the Python script 100 times
for i in {1..100}
do
    echo "Run #$i (Round $round, Model DSV3)"
    # python experiment_CD_MT2.py
    python experiment_CD_MT2.py --round $round
done