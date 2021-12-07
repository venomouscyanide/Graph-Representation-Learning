#!/usr/bin/env bash
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "karate" --identifier "node_2_vec_degree_information_karate" --model_name "node2vec" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "cora" --identifier "node_2_vec_degree_information_cora" --model_name "node2vec" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "citeseer" --identifier "node_2_vec_degree_information_citeseer" --model_name "node2vec" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "pubmed" --identifier "node_2_vec_degree_information_pubmed" --model_name "node2vec" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "computers" --identifier "node_2_vec_degree_information_computers" --model_name "node2vec" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "photo" --identifier "node_2_vec_degree_information_photo" --model_name "node2vec" --norm_features --degree_information

