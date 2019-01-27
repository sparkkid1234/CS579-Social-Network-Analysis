"""
sumarize.py
"""
import json 
import pickle

def read_user(filename):
	with open(filename,'rb') as f:
		users = pickle.loads(f.read())
	return users
def read_file(filename):
	with open(filename,'r') as f:
		info = json.load(f)
	return info
def main():
	users = read_user('users.txt')
	tweets = read_file('totaltweets.txt')
	graph_info = read_file('cluster.txt')
	info = read_file('classinfo.txt')
	
	with open('summary.txt','w') as file:
		file.write('Number of users collected: {}'.format(graph_info['nodes']))
		file.write('\nCollected a total of {} tweets'.format(len(tweets)))
		file.write('\nNumber of communities discovered: 2')
		file.write('\nNumber of users per community: ')
		for k,v in graph_info.items():
			if isinstance(v,list):
				file.write('\n\tIn community {}: {} users'.format(k.split('_')[1],len(v)))
		file.write('\nUsing a fined-tune Multinomial Naive Bayes on {} test tweets:'.format(info['test_set']))
		file.write('\n\tAccuracy on test set: {}'.format(str(info['accuracy'])))
		for k,v in info.items():
			if isinstance(v,list):
				file.write('\n\tNumber of instances for class {}: {}'.format(k.split('_')[1],len(v)))
		for k,v in info.items():
			if isinstance(v,str):
				file.write('\n\tExample of class {}: {}'.format(k.split('_')[1],v))
		
if __name__ == '__main__':
	main()