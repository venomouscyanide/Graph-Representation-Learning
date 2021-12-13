#!/usr/bin/env bash
python -m project.statistics.mmd_table_generation.deep_stats_table --base_folder deep_models --output_folder deep_table_clustering_exp --mmd_type "clustering"
python -m project.statistics.mmd_table_generation.deep_stats_table --base_folder deep_models --output_folder deep_table_deg_dist_exp --mmd_type "deg_distribution"
python -m project.statistics.mmd_table_generation.deep_stats_table --base_folder deep_models --output_folder deep_table_closeness_exp --mmd_type "closeness_centrality"
# Takes way too long
# python -m project.statistics.mmd_table_generation.deep_stats_table --base_folder deep_models --output_folder deep_table_betweeness_exp --mmd_type "betweeness_centrality"