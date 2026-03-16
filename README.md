# Thesis Implementation

This repository contains the implementation and experimental code for my Master's thesis.

The thesis investigates learning-based approaches for graph coloring using Graph Neural Networks (GNNs), with a focus on improving vertex ordering strategies for sparse derivative computation. The work will involve experimenting with graph structures, machine learning models, and comparisons with heuristic methods implemented in ColPack.

## Environment Setup

The project uses Python and Conda environments.

Create and activate the environment:

conda activate thesis

Install dependencies:

pip install -r requirements.txt

## Project Structure

data/  
- raw/ : original datasets and graph instances  
- processed/ : cleaned or transformed datasets  

notebooks/  
- experimental notebooks for exploration and visualization  

src/  
- models/ : neural network architectures  
- training/ : training scripts  
- graphs/ : graph processing utilities  
- utils/ : helper functions  

results/  
- experiment outputs and results  

configs/  
- configuration files for experiments  

references/  
- papers and supporting material