#!/usr/bin/env bash
echo "With norm and degree_information"
python -m project.non_structure_aware.run_experiments --dataset "karate" --identifier "mlp_norm_degree_information_karate" --use_norm --degree_information
python -m project.non_structure_aware.run_experiments --dataset "cora" --identifier "mlp_norm_degree_information_cora" --use_norm --degree_information
python -m project.non_structure_aware.run_experiments --dataset "citeseer" --identifier "mlp_norm_degree_information_citeceer" --use_norm --degree_information
python -m project.non_structure_aware.run_experiments --dataset "pubmed" --identifier "mlp_norm_degree_information_pubmed" --use_norm --degree_information
python -m project.non_structure_aware.run_experiments --dataset "computers" --identifier "mlp_norm_degree_information_computers" --use_norm --degree_information
python -m project.non_structure_aware.run_experiments --dataset "photo" --identifier "mlp_norm_degree_information_photo" --use_norm --degree_information
echo "With degree_information"
python -m project.non_structure_aware.run_experiments --dataset "karate" --identifier "mlp_no_norm_degree_information_karate" --degree_information
python -m project.non_structure_aware.run_experiments --dataset "cora" --identifier "mlp_no_norm_degree_information_cora" --degree_information
python -m project.non_structure_aware.run_experiments --dataset "citeseer" --identifier "mlp_no_norm_degree_information_citeceer" --degree_information
python -m project.non_structure_aware.run_experiments --dataset "pubmed" --identifier "mlp_no_norm_degree_information_pubmed" --degree_information
python -m project.non_structure_aware.run_experiments --dataset "computers" --identifier "mlp_no_norm_degree_information_computers" --degree_information
python -m project.non_structure_aware.run_experiments --dataset "photo" --identifier "mlp_no_norm_degree_information_photo" --degree_information
echo "With norm"
python -m project.non_structure_aware.run_experiments --dataset "karate" --identifier "mlp_norm_no_degree_information_karate" --use_norm
python -m project.non_structure_aware.run_experiments --dataset "cora" --identifier "mlp_norm_no_degree_information_cora" --use_norm
python -m project.non_structure_aware.run_experiments --dataset "citeseer" --identifier "mlp_norm_no_degree_information_citeceer" --use_norm
python -m project.non_structure_aware.run_experiments --dataset "pubmed" --identifier "mlp_norm_no_degree_information_pubmed" --use_norm
python -m project.non_structure_aware.run_experiments --dataset "computers" --identifier "mlp_norm_no_degree_information_computers" --use_norm
python -m project.non_structure_aware.run_experiments --dataset "photo" --identifier "mlp_norm_no_degree_information_photo" --use_norm
echo "without augmentations"
python -m project.non_structure_aware.run_experiments --dataset "karate" --identifier "mlp_no_norm_no_degree_information_karate"
python -m project.non_structure_aware.run_experiments --dataset "cora" --identifier "mlp_no_norm_no_degree_information_cora"
python -m project.non_structure_aware.run_experiments --dataset "citeseer" --identifier "mlp_no_norm_no_degree_information_citeceer"
python -m project.non_structure_aware.run_experiments --dataset "pubmed" --identifier "mlp_no_norm_no_degree_information_pubmed"
python -m project.non_structure_aware.run_experiments --dataset "computers" --identifier "mlp_no_norm_no_degree_information_computers"
python -m project.non_structure_aware.run_experiments --dataset "photo" --identifier "mlp_no_norm_no_degree_information_photo"