"""
cluster.py
"""
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import pickle
import numpy as np
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from collections import defaultdict
from itertools import combinations
import json

def get_users(filename):
	with open(filename,'rb') as f:
		users = pickle.loads(f.read())
	return users
	
def follower_count(users):
    c = Counter()
    for user in users:
        c.update(user['followers'])
    return c

def following_count(users):
    c = Counter()
    for user in users:
        c.update(user['following'])
    return c
	
def create_graph(users):
    """Create edge between following and followers
    Only add edge to followers who follow more than 2 seed users
    and following that is followed by more than 3 seed users
    """
    graph = nx.Graph()
    c = follower_count(users)
    c1 = following_count(users)
    for user in users:
        graph.add_node(user['id'])
        for friend in c1.items():
            if friend[1] >= 3:
                if friend[0] in user['following']:
                    graph.add_edge(int(user['id']),friend[0])
        for follower in c.items():
            if follower[1] >= 2:
                if follower[0] in user['followers']:
                    graph.add_edge(int(user['id']),follower[0])
    return graph

def recreate_graph(nodes,graph):
	new_graph = nx.Graph()
	node_pairs = combinations(nodes,2)
	for node1, node2 in node_pairs:
		if graph.has_edge(node1,node2):
			new_graph.add_edge(node1,node2)
	return new_graph

def relabel_node(graph,ids):
	"""Relabel 4 seeds user nodes from id to screen_name for drawing 
	since it's not possible to label some node string and the rest integer directly"""
	label = {}
	for node in graph.nodes():
		if node in ids.keys():
			label[node] = ids[node]
			
	return nx.relabel_nodes(graph,label)
			
def draw_network(graph,users,filename):
	ids = {user['id']:user['screen_name'] for user in users}
	relabeled_graph = relabel_node(graph,ids)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	label = {}
	color_map = []
	pos = nx.spring_layout(relabeled_graph,k = 0.09, weight = 0.2)
	seed_graph = relabeled_graph.copy()
	for node in graph.nodes():
		if node in ids.keys():
			label[ids[node]] = ids[node]
		else:
			seed_graph.remove_node(node)
			
	nx.draw_networkx(relabeled_graph, pos = pos, with_labels = False,node_size = 20,alpha = 0.5,width = 0.01, ax=ax,node_color = 'b')
	nx.draw_networkx_labels(relabeled_graph,pos=pos,labels = label,font_size=8, font_color = 'r',ax=ax)
	#Color the seed node
	relabeled_graph = seed_graph
	nx.draw_networkx_nodes(relabeled_graph,pos = pos,ax = ax,node_color = 'r',node_size = 80)
	plt.savefig(filename,dpi=300)

def volume(nodes,graph):
	edges = []
	for node in nodes:
		for neighbor in graph.neighbors(node):
			edges.append(tuple(sorted([neighbor,node])))
	edges = list(set(edges))
	return len(edges)
	
def cut(S,T,graph):
	count = 0
	edges = graph.edges()
	for node1, node2 in edges:
		if (node1 in S and node2 in T) or (node2 in S and node1 in T):
			count+=1
	return count
	
def norm_cut(S,T,graph):
	return (cut(S,T,graph)/(volume(S,graph))) + (cut(S,T,graph)/volume(T,graph))

def adjacency_matrix(graph):
	return nx.adjacency_matrix(graph,sorted(graph.nodes()))

def degree_matrix(graph):
	degrees = graph.degree().items()
	#Sort to be in the same order as adjacency_matrix
	degrees = sorted(degrees, key = lambda x: x[0])
	degrees = [d[1] for d in degrees]
	return np.diag(degrees)

def laplacian_matrix(graph):
	return degree_matrix(graph) - adjacency_matrix(graph)

def get_eigen(laplacian):
	eig_vals, eig_vectors = eigh(laplacian)
	return np.round(eig_vals,2), np.round(eig_vectors,2)

def cluster_eig(eig_vectors, nodes, method = 'total', theta = 0):
	nodes = sorted(nodes)
	first_comp = []
	second_comp = []
	components = []
	if method == 'total':
		eig_vectors = np.sum(eig_vectors, axis = 1)
		for i in range(len(nodes)):
			if eig_vectors[i] >= theta:
				first_comp.append(nodes[i])
			elif eig_vectors[i] < theta:
				second_comp.append(nodes[i])
	else:
		for i in range(len(nodes)):
			if eig_vectors[i][int(method)] >= theta:
				first_comp.append(nodes[i])
			elif eig_vectors[i][int(method)] < theta:
				second_comp.append(nodes[i])

	components.append(first_comp)
	components.append(second_comp)
	return components
	
def cluster_KMeans(eig_vectors, nodes):
	nodes = sorted(nodes)
	kmeans = KMeans(n_clusters = 2,random_state = 42)
	kmeans.fit(eig_vectors)
	labels = kmeans.labels_
	cluster = defaultdict(list)
	
	for i in range(len(labels)):
		cluster[labels[i]].append(nodes[i])
	return cluster

def print_cluster(eig_vectors,graph,method = 'total',theta = 0):
	print('Using Kmeans after eigen decomposition:')
	components = cluster_KMeans(eig_vectors,graph.nodes())
	for i in range(len(components)):
		print('\tCluster {} has {} nodes'.format(i+1,len(components[i])))
	print('\tNormalized cut value for this partition is: {}'.format(norm_cut(components[0],components[1],graph)))
	print('Using only eigen decomposition by totaling the eigenvectors:')
	components = cluster_eig(eig_vectors,graph.nodes())
	for i in range(len(components)):
		print('\tCluster {} has {} nodes'.format(i+1,len(components[i])))
	print('\tNormalized cut value for this partition is: {}'.format(norm_cut(components[0],components[1],graph)))
	components = cluster_eig(eig_vectors,graph.nodes(),method = '2')
	print('Using only eigen decomposition by considering the values of 2nd eigenvectors:')
	for i in range(len(components)):
		print('\tCluster {} has {} nodes'.format(i+1,len(components[i])))
	print('\tNormalized cut value for this partition is: {}'.format(norm_cut(components[0],components[1],graph)))
	
	return components

def main():
    users = get_users('users.txt')
    print('Getting 4 seed users:'+" "+', '.join([user['screen_name'] for user in users]))
    print('Creating graph...')
    graph = create_graph(users)
    print('Saving graph to file...')
    draw_network(graph,users,'network1.png')
    print('Graph has {} nodes and {} edges'.format(len(graph.nodes()),len(graph.edges())))
    matrix = laplacian_matrix(graph)
    eig_vals, eig_vectors = get_eigen(matrix)
    print('\nClustering graph into 2 components...')
    components = print_cluster(eig_vectors,graph)
    print('\nDrawing 2 new clusters by considering 2nd eigenvectors method into file cluster...')
    new_graph = nx.compose(recreate_graph(components[0],graph),recreate_graph(components[1],graph))
    draw_network(new_graph,users,'clusters.png')
    graph_info = {'nodes':len(graph.nodes()),'edges':len(graph.edges()),'cluster_1':components[0],'cluster_2':components[1]}
    with open('cluster.txt','w') as file:
	    json.dump(graph_info,file)
	
if __name__== '__main__':
	main()