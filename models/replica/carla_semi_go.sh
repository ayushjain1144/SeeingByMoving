#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_SEMI GO"
echo "-----"

MODE="CARLA_SEMI"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_SEMI GO DONE"
echo "----------"

