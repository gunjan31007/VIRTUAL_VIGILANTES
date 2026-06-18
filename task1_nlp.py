# Task I - Natural Language Processing (NLP) on Text Data
# Virtual Vigilantes Data Science Internship
# Libraries: NLTK, TextBlob, Matplotlib, WordCloud

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud
import string
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. SAMPLE DATASET
#    A small collection of product reviews
# ─────────────────────────────────────────
reviews = [
    "I absolutely love this product! It works perfectly and exceeded all my expectations.",
    "Terrible experience. The item broke after just two days. Very disappointed.",
    "Decent quality for the price. Nothing extraordinary but it does the job.",
    "Amazing! Fast delivery and great packaging. Will definitely buy again.",
    "Not worth the money. Poor build quality and the customer service was unhelpful.",
    "Pretty good overall. The design looks nice and setup was easy.",
    "Worst purchase I have ever made. Complete waste of money.",
    "Fantastic product! Highly recommend to anyone looking for value and performance.",
    "It is okay. Not too bad, not too great. Just average.",
    "Superb quality! Works exactly as described. Five stars from me.",
    "Disappointed with the product. It stopped working after a week.",
    "Excellent value for money. I am very happy with my purchase.",
    "The product feels cheap. Expected much better quality at this price.",
    "Great product! My whole family loves it. Will buy again for sure.",
    "Below average. The product does not match the description at all.",
]

df = pd.DataFrame({'review': reviews})

print("=" * 55)
print("   TASK I — NLP ON TEXT DATA")
print("   Virtual Vigilantes Data Science Internship")
print("=" * 55)

# ─────────────────────────────────────────
# 2. TOKENIZATION
# ─────────────────────────────────────────
print("\n[1] TOKENIZATION")
print("-" * 40)

# Word tokenize the first review as a demo
sample = reviews[0]
word_tokens  = word_tokenize(sample)
sent_tokens  = sent_tokenize(sample)

print(f"Original  : {sample}")
print(f"Words     : {word_tokens}")
print(f"Sentences : {sent_tokens}")

# ─────────────────────────────────────────
# 3. STOPWORD REMOVAL & FREQUENCY COUNT
# ─────────────────────────────────────────
print("\n[2] STOPWORD REMOVAL & WORD FREQUENCY")
print("-" * 40)

stop_words = set(stopwords.words('english'))
all_words  = []

for review in reviews:
    tokens = word_tokenize(review.lower())
    # Keep only alphabetic tokens that are not stopwords
    filtered = [w for w in tokens
                if w.isalpha() and w not in stop_words]
    all_words.extend(filtered)

freq = Counter(all_words)
top10 = freq.most_common(10)
print("Top 10 most frequent meaningful words:")
for word, count in top10:
    print(f"  {word:<15} {count}")

# ─────────────────────────────────────────
# 4. SENTIMENT ANALYSIS (VADER)
# ─────────────────────────────────────────
print("\n[3] SENTIMENT ANALYSIS (VADER)")
print("-" * 40)

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment']       = df['review'].apply(get_sentiment)
df['compound_score']  = df['review'].apply(
    lambda x: sia.polarity_scores(x)['compound'])

print(df[['review', 'sentiment', 'compound_score']].to_string(index=False))

# Sentiment counts
counts = df['sentiment'].value_counts()
print(f"\nSentiment Distribution:\n{counts.to_string()}")

# ─────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────
print("\n[4] GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Task I — NLP on Text Data\nVirtual Vigilantes Data Science Internship",
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Sentiment Distribution (Pie) ──
ax1 = fig.add_subplot(gs[0, 0])
colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}
pie_colors = [colors[s] for s in counts.index]
ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
        colors=pie_colors, startangle=140,
        textprops={'fontsize': 11})
ax1.set_title("Sentiment Distribution", fontweight='bold')

# ── Plot 2: Top 10 Words (Bar) ──
ax2 = fig.add_subplot(gs[0, 1])
words_list  = [w for w, _ in top10]
counts_list = [c for _, c in top10]
bars = ax2.barh(words_list[::-1], counts_list[::-1],
                color='steelblue', edgecolor='white')
ax2.set_xlabel("Frequency")
ax2.set_title("Top 10 Most Frequent Words", fontweight='bold')
for bar, val in zip(bars, counts_list[::-1]):
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)
ax2.set_xlim(0, max(counts_list) + 1)

# ── Plot 3: Compound Score per Review ──
ax3 = fig.add_subplot(gs[1, 0])
bar_colors = [colors[s] for s in df['sentiment']]
ax3.bar(range(1, len(df)+1), df['compound_score'],
        color=bar_colors, edgecolor='white')
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax3.axhline(0.05,  color='green', linewidth=0.6, linestyle=':')
ax3.axhline(-0.05, color='red',   linewidth=0.6, linestyle=':')
ax3.set_xlabel("Review #")
ax3.set_ylabel("Compound Score")
ax3.set_title("Sentiment Score per Review", fontweight='bold')
ax3.set_xticks(range(1, len(df)+1))

# Legend
from matplotlib.patches import Patch
legend = [Patch(color='#2ecc71', label='Positive'),
          Patch(color='#e74c3c', label='Negative'),
          Patch(color='#f39c12', label='Neutral')]
ax3.legend(handles=legend, loc='lower right', fontsize=8)

# ── Plot 4: Word Cloud ──
ax4 = fig.add_subplot(gs[1, 1])
text_for_cloud = ' '.join(all_words)
wc = WordCloud(width=600, height=400, background_color='white',
               colormap='RdYlGn', max_words=80).generate(text_for_cloud)
ax4.imshow(wc, interpolation='bilinear')
ax4.axis('off')
ax4.set_title("Word Cloud (after stopword removal)", fontweight='bold')

plt.savefig('/mnt/user-data/outputs/task1_nlp_output.png',
            dpi=150, bbox_inches='tight')
print("Saved: task1_nlp_output.png")
plt.close()

print("\n[DONE] Task I complete.")
