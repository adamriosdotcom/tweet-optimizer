import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# load data from your sqlite database
conn = sqlite3.connect('tweets.db')
df = pd.read_sql_query("SELECT * FROM tweets", conn)
conn.close()

# check the dataframe structure
print(df.info())
print(df.describe())

# convert key columns to numeric (handling errors gracefully)
numeric_cols = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount', 'viewCount', 'bookmarkCount']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# visualize the distribution of likes
plt.hist(df['likeCount'].dropna(), bins=50)
plt.title('Distribution of Like Counts')
plt.xlabel('Likes')
plt.ylabel('Frequency')
plt.show()
