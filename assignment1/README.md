# Assignment 1 
Assignment 1 part of the GRL course led by [Dr. Amirali Salehi-Abari](https://www.abari.ca/)


## Problem Definition
- Create node vectors which contain statistical properties of the node such as degree, various centralities (e.g., closeness, eigen centrality, betweenness), etc. Each dimension captures a unique property. 
    
    - Train a GNN to automatically generate node embeddings such that each learn node embedding can accurately predict the node statistical property vector. Hint, for training GNNs, you might need to put multiple regressor heads on top of GNNs while minimizing the loss between regressors’ predictions and node properties vectors.
    - Run experiments on multiple medium-size graphs and a few node embedding techniques.  

