#!/usr/bin/env bash
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 1 --dataset "karate" --identifier "dw_no_degree_information_karate" --model_name "deepwalk" --norm_features
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 1 --dataset "cora" --identifier "dw_no_degree_information_cora" --model_name "deepwalk" --norm_features
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 1 --dataset "citeseer" --identifier "dw_no_degree_information_citeseer" --model_name "deepwalk" --norm_features
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 1 --dataset "pubmed" --identifier "dw_no_degree_information_pubmed" --model_name "deepwalk" --norm_features
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 1 --dataset "computers" --identifier "dw_no_degree_information_computers" --model_name "deepwalk" --norm_features
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 1 --dataset "photo" --identifier "dw_no_degree_information_photo" --model_name "deepwalk" --norm_features
