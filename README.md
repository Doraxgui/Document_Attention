# Overview
This repository is a system about [SemEval-2025 Task 9: The Food Hazard Detection Challenge](https://food-hazard-detection-semeval-2025.github.io/)

This is a classification task, and its challenges lie in: 
  1) data imbalance
  2) excessive number of labels
     
This repository provides a solution based on LLM, specific system description can be reffered [here].

# Getting Started
## Installation
```
git clone https://github.com/Doraxgui/Document_Attention.git
cd Document_Attention
pip install -r requirements.txt
```
## Convert Data Format
Download the official dataset and run these commands
```
cd Document_Attention/data/data
python data_gerenate.py
```
## Generate Embeddings
```
cd Document_Attention/data/embeddings
python embedding_generate.py
```
##  Train Model
```
cd Document_Attention/sft
sh launch.sh
```
