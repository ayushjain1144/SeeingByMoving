#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_VIEWMINE GO"
echo "-----"

MODE="CARLA_VIEWMINE"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_VIEWMINE GO DONE"
echo "----------"

