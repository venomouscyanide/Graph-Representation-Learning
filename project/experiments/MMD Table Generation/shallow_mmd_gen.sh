#!/usr/bin/env bash
python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_exp --mmd_type "clustering"
python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_exp --mmd_type "deg_distribution"
python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_exp --mmd_type "betweeness_centrality"
python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_exp --mmd_type "closeness_centrality"