import networkx as nx


def explore():
    G = nx.Graph()
    G.add_node(0, feature=0, label=0)

    # Get attributes of the node 0
    node_0_attr = G.nodes[0]
    print("Node 0 has the attributes {}".format(node_0_attr))

    # Add multiple nodes with attributes
    G.add_nodes_from([
        (1, {"feature": 1, "label": 1}),
        (2, {"feature": 2, "label": 2})
    ])

    # Loop through all the nodes
    # Set data=True will return node attributes
    for node in G.nodes(data=True):
        print(node)

    for node in G.nodes(data=True):
        print(node)

    # Get number of nodes
    num_nodes = G.number_of_nodes()
    print("G has {} nodes".format(num_nodes))


if __name__ == '__main__':
    explore()
