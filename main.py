import time

from datasets import load_dataset

ds = load_dataset("lewtun/music_genres")

train = ds['train']
train = train.shuffle(seed=int(time.time()))
# genres = train['genre_id']


