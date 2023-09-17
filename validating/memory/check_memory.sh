#!/bin/bash

# Print header
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv | head -n 1

# Loop
for i in {1..1000}; do
  nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv | tail -n 1
  sleep 5
done