#!/usr/bin/env bash
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "karate" --identifier "dw_degree_information_karate" --model_name "deepwalk" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "cora" --identifier "dw_degree_information_cora" --model_name "deepwalk" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "citeseer" --identifier "dw_degree_information_citeseer" --model_name "deepwalk" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "pubmed" --identifier "dw_degree_information_pubmed" --model_name "deepwalk" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "computers" --identifier "dw_degree_information_computers" --model_name "deepwalk" --norm_features --degree_information
python -m project.shallow_models.model_trainer --cpu_count 1 --gpu_count 0 --dataset "photo" --identifier "dw_degree_information_photo" --model_name "deepwalk" --norm_features --degree_information

