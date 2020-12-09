import praw
import progressbar
import pandas as pd

reddit = praw.Reddit(
    client_id="OYmafsxlFa1yFQ",
    client_secret="JoesS3fgmi18eyCom3t0B-l1eKw6tQ",
    user_agent="Reddit Scraping"
)


def grab_top_and_hot(sr, top_lim=2000, hot_lim=1000):
    database = []
    bar = progressbar.ProgressBar(max_value=7000)
    progress = 0
    print('Now scraping the subreddit:' + sr)
    for i, post in enumerate(reddit.subreddit(sr).top(time_filter='all', limit=top_lim)):
        database.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments])
        progress += 1
        bar.update(progress)
    for i, post in enumerate(reddit.subreddit(sr).top(time_filter='month', limit=top_lim)):
        database.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments])
        progress += 1
        bar.update(progress)
    for i, post in enumerate(reddit.subreddit(sr).top(time_filter='year', limit=top_lim)):
        database.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments])
        progress += 1
        bar.update(progress)
    for i, post in enumerate(reddit.subreddit(sr).new(limit=hot_lim)):
        database.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments])
        progress += 1
        bar.update(progress)
    return database


def print_to_csv(data, name):
    data = pd.DataFrame(data, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments'])
    data.to_csv(str(name + '.csv'), index=False)


def remove_duplicates(data):
    s = set()
    new_db = []
    for post in data:
        if post[0] in s:
            continue
        else:
            new_db.append(post)
        s.add(post[0])
    return new_db


def scrape(subreddits=None):
    if subreddits is None:
        subreddits = ['politics', 'uspolitics', 'americanpolitics', 'conservative']
    data = []
    for sr in subreddits:
        temp = grab_top_and_hot(sr)
        data.extend(temp)

    data = remove_duplicates(data)
    # shuffles the rows in the combined dataframe
    data = data.sample(frac=1)
    data.to_csv('data.csv', index=False)




