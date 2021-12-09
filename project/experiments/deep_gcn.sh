#!/usr/bin/env bash
echo "With norm and degree_information"
python -m project.deep_models.deep_trainer --dataset "karate" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_degree_information_karate" --use_norm --degree_information
python -m project.deep_models.deep_trainer --dataset "cora" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_degree_information_cora" --use_norm --degree_information
python -m project.deep_models.deep_trainer --dataset "citeseer" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_degree_information_citeceer" --use_norm --degree_information
python -m project.deep_models.deep_trainer --dataset "pubmed" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_degree_information_pubmed" --use_norm --degree_information
python -m project.deep_models.deep_trainer --dataset "computers" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_degree_information_computers" --use_norm --degree_information
python -m project.deep_models.deep_trainer --dataset "photo" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_degree_information_photo" --use_norm --degree_information
echo "With norm"
python -m project.deep_models.deep_trainer --dataset "karate" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_no_norm_degree_information_karate" --degree_information
python -m project.deep_models.deep_trainer --dataset "cora" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_no_norm_degree_information_cora" --degree_information
python -m project.deep_models.deep_trainer --dataset "citeseer" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_no_norm_degree_information_citeceer" --degree_information
python -m project.deep_models.deep_trainer --dataset "pubmed" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_no_norm_degree_information_pubmed" --degree_information
python -m project.deep_models.deep_trainer --dataset "computers" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_no_norm_degree_information_computers" --degree_information
python -m project.deep_models.deep_trainer --dataset "photo" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_no_norm_degree_information_photo" --degree_information
echo "With degree_information"
python -m project.deep_models.deep_trainer --dataset "karate" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_no_degree_information_karate" --use_norm
python -m project.deep_models.deep_trainer --dataset "cora" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_no_degree_information_cora" --use_norm
python -m project.deep_models.deep_trainer --dataset "citeseer" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_no_degree_information_citeceer" --use_norm
python -m project.deep_models.deep_trainer --dataset "pubmed" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_no_degree_information_pubmed" --use_norm
python -m project.deep_models.deep_trainer --dataset "computers" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_no_degree_information_computers" --use_norm
python -m project.deep_models.deep_trainer --dataset "photo" --gpu_count 0 --cpu_count 1 --model_name "gcn" --identifier "gcn_norm_no_degree_information_photo" --use_norm