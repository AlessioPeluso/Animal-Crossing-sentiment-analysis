
# --------------------------------------------------------------------------------------------------

# -------------------------------------
# -- Animal Crossing sentiment analysis
# -------------------------------------

# How do negative and positive reviews compare in the words they use?

# --------------------------------------------------------------------------------------------------


# import packages
import pandas as pd
import numpy as np
import os

import itertools
import collections
import nltk
from nltk.corpus import stopwords
import re

from shifterator import relative_shift as rs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# load the data
reviews = pd.read_csv("Natural_Language/Animal_Crossing/data/critic.csv", encoding='utf-8')

reviews.head()

# --------------------------------------------------------------------------------------------------

# Data visualization

# average review grades over time
reviews['date'] = pd.to_datetime(reviews['date'])
reviews.index = reviews['date']

mean_daily_grades = reviews.resample('D', on='date').mean().reset_index('date')

# plot horizontal bar graph
fig, ax = plt.subplots(figsize=(12, 8))
monthly_plot = sns.lineplot(data = mean_daily_grades,
                      x = 'date',
                      y = 'grade',
                      color="navy"
                      )

ax.set_title("Average daily grade")
x_dates = mean_daily_grades['date'].dt.strftime('%m-%d').sort_values().unique()
ax.set_xticklabels(labels=x_dates, rotation=45, ha='right')

plt.show()

# --------------------------------------------------------------------------------------------------

# Data cleaning and preparation

# divide reviews into positive and negative based on the median grade
median_grade = reviews.grade.median()

reviews.loc[reviews['grade'] <= median_grade, 'review_category'] = 'Negative'
reviews.loc[reviews['grade'] > median_grade, 'review_category'] = 'Positive'

# divide in two datasets
reviews_neg = reviews[reviews['review_category'] == 'Negative']
reviews_pos = reviews[reviews['review_category'] == 'Positive']

# divide only the texts
texts = reviews['text'].tolist()
texts_neg = reviews_neg['text'].tolist()
texts_pos = reviews_pos['text'].tolist()

# to remove stop words
stop_words = set(stopwords.words('english'))

%run Natural_Language/Animal_Crossing/helpful_functions.py

# clean up the review texts
clean_texts = clean_text(texts)
clean_texts_neg = clean_text(texts_neg)
clean_texts_pos = clean_text(texts_pos)

# --------------------------------------------------------------------------------------------------

# Data visualization 

# dataframes for most frequent common words in positive and negative reviews
common = pd.DataFrame(clean_texts.most_common(15),
                             columns=['words', 'count'])
common_neg = pd.DataFrame(clean_texts_neg.most_common(15),
                             columns=['words', 'count'])
common_pos = pd.DataFrame(clean_texts_pos.most_common(15),
                             columns=['words', 'count'])

# plot horizontal bar graph
fig, ax = plt.subplots(figsize=(8, 8))
common_neg.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="red")

ax.set_title("Common Words Found in Negative Reviews")
plt.show()

# Plot horizontal bar graph
fig, ax = plt.subplots(figsize=(8, 8))
common_pos.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="green")
ax.set_title("Common Words Found in Positive Reviews")
plt.show()

# word cloud for all the reviews
#!pip install wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.figure(figsize = (20,20))
wordcloud = WordCloud(max_words = 1000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(texts))
plt.imshow(wordcloud , interpolation = 'bilinear')

# NPL ART
animal = 'https://ih0.redbubble.net/image.1086694087.7585/ap,550x550,12x12,1,transparent,t.u2.png'
with urllib.request.urlopen(animal) as url:
    f = BytesIO(url.read())
img = Image.open(f)

mask = np.array(img)
img_color = ImageColorGenerator(mask)

wordcloud = WordCloud(background_color='white', mask=mask, max_font_size=2000, max_words=2000, random_state=1612, stopwords = STOPWORDS).generate(" ".join(texts))

plt.figure(figsize=(16, 10))
# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor
plt.axis('off')
plt.imshow(wordcloud.recolor(color_func=img_color), interpolation="bilinear")
plt.show()
# --------------------------------------------------------------------------------------------------
