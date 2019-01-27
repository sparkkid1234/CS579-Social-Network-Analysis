"""
collect.py
"""
import networkx as nx
from TwitterAPI import TwitterAPI
import sys
import time
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import json

consumer_key = 'W7KIWnCiefaD5S642hzyDkgV1'
consumer_secret = 'zuzZL3CR0XgF3RRfI4hanJYoz6miG33lPjcnSLXqXto8Bjd8MO'
access_token = '812945826360111104-w73ivervI9bkOSg4ESgWEHFbOdV4ZsP'
access_token_secret = 'eXOR3J6KLBNIpEhAnTll0TfGvfJanoY9FAd5bISG1jacr'

def remove_nonprintable(texts):
	"""Remove all emojis etc, replace them with a space"""
	new_texts= []
	for text in texts:
		new_texts.append(''.join(i for i in text if ord(i) < 128))
	return new_texts
	
def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

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
            
def get_user(twitter,names):
    return list(robust_request(twitter,'users/lookup',{'screen_name':names}))

def get_tweet(twitter,name,count = 200):
	"""Getting the FULL UNTRUNCATED TWEETS. Doesnt work with twitterapi sadly"""
	params = {'screen_name':name,'include_rts':False,'count':count}
	texts = []
	for tweet in robust_request(twitter, 'statuses/user_timeline',params):
		if 'text' in tweet and tweet['truncated'] == False:
			texts.append(tweet['text'])
		if 'text' in tweet and tweet['truncated'] == True:
			#texts.append(tweet['extended_tweet']['full_text'])
			texts.append(tweet['text'])
	return texts

def get_all_tweets(twitter, users):
	tweets = []
	for user in users:
		tweets.extend(get_tweet(twitter,user['screen_name']))
	return tweets
	
def get_follower(twitter,user_name):
    return [ids for ids in robust_request(twitter,'followers/ids',{'screen_name':user_name,
                                                               'count':5000})]
def get_following(twitter,user_name):
    return [ids for ids in robust_request(twitter,'friends/ids',{'screen_name':user_name,
                                                               'count':5000})]
def get_all_following(twitter,users):
    for user in users:
        user['following'] = get_following(twitter,user['screen_name'])

def get_all_followers(twitter,users):
    for user in users:
        user['followers'] = get_follower(twitter,user['screen_name'])
	
def get_all_friends(twitter,users):
    """Get all the people that follow that user and being followed by the user"""
    get_all_following(twitter, users)
    get_all_followers(twitter, users)
    for user in users:
        user['friends'] = [ids for ids in set(user['following']) & set(user['followers'])]

def get_name_from_id(twitter,ids):
    ids_list = [key for key in ids.keys()]
    users = [user for user in 
                 robust_request(twitter,'users/lookup',{'user_id':ids_list})]
    user_name = {}
    for i in range(len(ids_list)):
        user_name[users[i]['screen_name']] = ids[users[i]['id']]
    return user_name

def friend_count(users):
    c = Counter()
    for user in users:
        c.update(user['friends'])
    return c

def print_num_follower(users):
    for user in users:
        print('\t{} has {} followers'.format(user['screen_name'],len(user['followers'])))

def print_num_following(users):
    for user in users:
        print('\t{} is following {} users'.format(user['screen_name'],len(user['following'])))

def print_num_tweets(users):
	for user in users:
		print('\t{} tweets collected for user {}'.format(len(user['tweets']),user['screen_name']))

def print_first_tweet(users):
	#for user in users:
		#print('\tUser {} first tweet: {}'.format(user['screen_name'],str(user['tweets'][0])))
	print('\tUser {} first tweet: '.format(users['screen_name'])+str(users['tweets'][0]))
	
def save_users(filename, users):
	with open(filename,'wb') as f:
		pickle.dump(users,f)

def save_tweets(filename,tweets):
	with open(filename,'w') as f:
		json.dump(tweets,f)

def main():
    twitter = get_twitter()
    print('Twitter Connection Established')
    names = ['akstanwyck','NikkiFinke','slashfilm','ErikDavis']
    print('Getting 4 seed users:'+" "+', '.join(names))
    users = get_user(twitter, names)
    get_all_friends(twitter, users)
    print('\nNumber of followers for each seed users: ')
    print_num_follower(users)
    print('\nNumber of following for each seed users: ')
    print_num_following(users)
    print("\nRetrieving all 4 users' friends, that is people who follow and being followed by each of the five:")
    for user in users:
        print('\tUser {} has {} friends'.format(user['screen_name'],len(user['friends'])))
    counter = friend_count(users)
    print('\n3 most common friends: ')
    ids = {c[0] : c[1] for c in counter.most_common(3)}
    user_name = get_name_from_id(twitter, ids)
    for name,count in user_name.items():
        print('\t{} is followed by and is following {} users'.format(name,count))
    male_user = [user for user in users if user['screen_name'] == 'slashfilm' or user['screen_name'] == 'ErikDavis']
    female_user = [user for user in users if user['screen_name'] == 'akstanwyck' or user['screen_name'] == 'NikkiFinke']
    male_tweets = get_all_tweets(twitter,male_user)
    female_tweets = get_all_tweets(twitter,female_user)
    print('\nRetrieving {} tweets for 2 male seed users'.format(len(male_tweets)))
    print('Retrieving {} tweets for 2 female seed users'.format(len(male_tweets)))
    save_users('users.txt',users)
    save_tweets('maletweets.txt',male_tweets)
    save_tweets('femaletweets.txt',female_tweets)
	
if __name__ == '__main__':
    main()
    