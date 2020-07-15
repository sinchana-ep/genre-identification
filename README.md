# Genre Identification on (a sub-set of) Gutenberg Corpus

This project presents a model that classifies a fiction book by its genre, using selected 19th century books from [‘Project Gutenberg’](https://www.gutenberg.org/). The task is a supervised multi-class text classification problem for the detection of genre of a book. This dataset is a [subset](http://dke.ovgu.de/findke/en/Research/Data+Sets-p-1140.html) of the Gutenburg corpus consisting of 1079 books belonging to 9 different genres. Hand-crafted feature representation was performed for this task. Each book that belongs to a particular genre belongs to that genre as it has some distinguishing characteristics. This idea was exploited to extract features to make the model understand the style, setting, sentiment, and plot of each book. In simple words, to understand, and answer the question, what features make each book belong to that genre?. The usual machine learning pipeline of data pre-processing, modeling, and evaluation was followed to solve this task. This project is implemented using Python 3, latest NLTK (v3.5) library, scikit-learn v0.23, and the imbalanced-learn API.

The table below illustrates the class distribution in the dataset.

| Genre         | Number of books/instances| 
| ------------- |:-------------:     |
| Literary     | 794     | 
| Detective and Mystery      | 111          |  
| Sea and Adventure | 36          |  
| Love and Romance | 18          |
| Western Stories | 18          |
| Ghost and Horror | 6          |
| Humorous and Wit and Satire | 6          |
| Christmas Stories | 5          |
| Allegories | 2         |

Hand-crafted feature extraction includes:

POS Tagging: nltk.pos_tag()

Named Entity Recognition:  nltk.ne_chunk()

Sentiment Analysis: SentimentIntensityAnalyzer().polarity_scores()(from nltk.sentiment.vader)

The table below gives an overview of the hand-crafted features. 

| Feature type |Feature|
| ------------- |:-------------:|
| 1. Writing style |Paragraph count, Female pronoun count, Male pronoun count, Semi colon count, Average sentence length, All POS tags|
| 2. Sentence Complexity | Average sentence length, POS tags (Comma, Period, Punctuation, Conjunction)|
|3. Female oriented | Female pronoun count|
|4. Male oriented | Male pronoun count|
| 5. Setting | POS tags (Quotes), Persons count, Locations count|
| 6. Sentiment | Compound score, Negative, Neutral, Positive|
|7. Plot complexity | Persons count, Locations count|

# Challenges

There is a huge class imbalance issue with the data set. The class 'Literary' dominates over the other classes with 794 instances, and the class 'Allegories' is on the other end of the spectrum with only 2 instances. Class imbalance is an issue because it gives a low predictive accuracy over the imbalanced classes. Using the imbalanced data set will likely predict every instance as the majority class and give a high accuracy result, which is not useful for our task at all. This problem is addressed during the implementation stage by using the correct evaluation metrics and over-sampling strategies (we have used [SMOTE](https://arxiv.org/abs/1106.1813)).

Another challenge was to carefully select and extract the right features. Selecting the right features is extremely important in building the correct machine learning model. In this project, it was important to know what features were beneficial in classifying each book to its correct genre.  

Key words—English Fiction, Text classification, Feature extraction, Gutenberg
