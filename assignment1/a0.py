# coding: utf-8

"""
Given a list of Twitter accounts of 4
U.S. presedential candidates from the previous election.

The goal is to use the Twitter API to construct a social network of these
accounts. We will then use the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

1. Create an account on [twitter.com](http://twitter.com).
2. Generate authentication tokens by following the instructions [here](https://dev.twitter.com/docs/auth/tokens-devtwittercom).
3. Add your tokens to the key/token variables below. (API Key == Consumer Key)
4. Be sure you've installed the Python modules
[networkx](http://networkx.github.io/) and
[TwitterAPI](https://github.com/geduldig/TwitterAPI). Assuming you've already
installed [pip](http://pip.readthedocs.org/en/latest/installing.html)

Output in log.txt file
"""

# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = 'W7KIWnCiefaD5S642hzyDkgV1'
consumer_secret = 'zuzZL3CR0XgF3RRfI4hanJYoz6miG33lPjcnSLXqXto8Bjd8MO'
access_token = '812945826360111104-w73ivervI9bkOSg4ESgWEHFbOdV4ZsP'
access_token_secret = 'eXOR3J6KLBNIpEhAnTll0TfGvfJanoY9FAd5bISG1jacr'


# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    file = open('candidates.txt','r')
    return [line[:-1] if line[-1] == '\n' else line for line in file]


def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    
    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    #return twitter.request('users/lookup',{'screen_name':screen_names})
    return robust_request(twitter, 'users/lookup', {'screen_name':screen_names}, max_tries=5)


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    #return [ids for ids in twitter.request('friends/ids',{'screen_name':screen_name,'count':5000})]
    return [ids for ids in robust_request(twitter, 'friends/ids',{'screen_name':screen_name,'count':5000}, max_tries=5)]

def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    for user in users:
        user['friends'] = get_friends(twitter,user['screen_name'])

def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    for user in users:
        print('%s %d' % (user['screen_name'],len(user['friends'])))


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    counter = Counter()
    for user in users:
        counter.update(user['friends'])
    return counter


def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.

    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    overlap_list = []
    counters = []
    
    for user in users:
        counters.append(Counter(user['friends']))
    for i in range(0,len(counters)-1):
        for j in range(i+1,len(counters)):
            overlap_list.append((users[i]['screen_name'],users[j]['screen_name'],
                       len(list((counters[i]&counters[j]).elements()))))
    #We can sort in order like this since python sorting is STABLE
    #Sort by user 2's screen_name first since it's the last dependency to break the tie
    overlap_list = sorted(overlap_list,key = lambda u: u[1])
    #Then sort by user 1's name since it's the second dependency to break the tie
    overlap_list = sorted(overlap_list,key = lambda u: u[0])
    #Then sort by number of follower since it's the primary sorting requirement
    overlap_list = sorted(overlap_list,key = lambda u: u[2],reverse = True)
    
    #This sorting sequence will assure the ties being broken the way the prompt wants
    return overlap_list    


def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump.

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    #Extract only info of hillary and donald trump
	#<!!!!> clone the list, else users list will be changed
    hillary_donald = list(users)

    for user in hillary_donald:
        if user['screen_name'] != 'HillaryClinton' or user['screen_name'] != 'realDonaldTrump':
            hillary_donald.remove(user)
    #Find the friend both follows
    counter1 = Counter(hillary_donald[0]['friends'])
    counter2 = Counter(hillary_donald[1]['friends'])
    common_friend_id = list((counter1 & counter2).elements())[0]
    
    #Get name by ID
    common_friend = [friend for friend in 
                     robust_request(twitter, 'users/lookup', {'user_id':common_friend_id}, max_tries=5)]
    return common_friend[0]['screen_name']
    

def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    graph = nx.Graph()
    for user in users:
        graph.add_node(user['screen_name'])

    for item in friend_counts.items():
        if item[1] >=2:
            graph.add_node(item[0])
            for user in users:
                if item[0] in user['friends']:
                    graph.add_edge(user['screen_name'],item[0])
    
    return graph
    

def draw_network(graph, users, filename):
	candidate = [user['screen_name'] for user in users]
	labels = {}
	for node in graph.nodes():
		if node in candidate:
			labels[node] = node
			
	plt.figure(figsize = (10,10))
	nx.draw_networkx(graph,with_labels = False, node_size = 100, alpha = 0.3)
	nx.draw_networkx_labels(graph,nx.spring_layout(graph),labels=labels,font_size = 12)
	plt.savefig(filename)


def main():
     #Main method. You should not modify this.
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()