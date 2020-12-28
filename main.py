import string
from collections import Counter

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

text = open('read.txt', encoding='utf-8').read()
lowerCase = text.lower()
cleanText = lowerCase.translate(str.maketrans('', '', string.punctuation))

# Using word_tokenize because it's faster than split()
tokens = word_tokenize(cleanText, "english")

# Removing Stop Words
finalWords = []
for word in tokens:
    if word not in stopwords.words('english'):
        finalWords.append(word)

# Lemmatization - From plural to single + Base form of a word
lemmaWords = []
for word in finalWords:
    word = WordNetLemmatizer().lemmatize(word)
    lemmaWords.append(word)

#print(lemmaWords)

emotionList = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clearLine = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clearLine.split(':')
        for word in lemmaWords:
            emotionList.append(emotion)

w = Counter(emotionList)
print(w)


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")


sentiment_analyse(cleanText)

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()