#!/usr/bin/env bash
python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_clustering_exp --mmd_type "clustering"
python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_deg_dist_exp --mmd_type "deg_distribution"
python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_closeness_exp --mmd_type "closeness_centrality"
# Takes way too long
# python -m project.statistics.mmd_table_generation.shallow_stats_table --base_folder shallow_models --output_folder shallow_table_betweeness_exp --mmd_type "betweeness_centrality"