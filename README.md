# Sentiment_Analysis_Financial_Tweets

## DATA

The data we use comes from HuggingFace "zeroshot/twitter-financial-news-sentiment"
The dataset holds 11,932 documents annotated with 3 labels:
* Bearish = 0
* Bullish = 1
* Neutral = 2

## Goals

Assess the prediction performance on sentiment classification task on financial tweets

### Naive model

We will apply a naïve model to predict the sentiment score of set of tweets, To do so, we will use a dictionary `SentiWordNet`
We will improve our model by taking into account of potential emoji and negation and booster words in the tweets.

### Vader Algorithm

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a pre-built, lexicon and rule-based sentiment analysis tool designed for analyzing text data in natural language. It is specifically crafted to handle sentiments expressed in social media texts, as it incorporates features like handling of emoticons, capitalization, and context-based sentiment scoring.

Here are some key aspects of the VADER library:

Lexicon and Rule-Based Approach: VADER uses a combination of a sentiment lexicon (a predefined list of words and their associated sentiment scores) and a set of grammatical and syntactical rules to determine the sentiment of a piece of text.

* Valence Scores: The lexicon assigns polarity scores to words, indicating the positive or negative sentiment conveyed by each word. These scores range from -1 to 1, where -1 represents extreme negativity, 1 represents extreme positivity, and 0 represents neutrality.

* Emoticon Handling: VADER is designed to handle sentiments expressed through emoticons, making it suitable for analyzing text data from social media platforms where emoticons are commonly used to convey emotions.

* Capitalization and Punctuation: VADER takes into account the intensity of sentiment by considering the impact of capitalization and punctuation in the text.

* Contextual Valence Shifting: VADER can recognize and handle some degree of valence shifting, where the sentiment of a word changes based on the context in which it is used.

* Sentiment Intensity: VADER provides a compound score that represents the overall sentiment intensity of a piece of text. This score considers both the individual word scores and their arrangement in the text.

VADER is implemented in Python and is part of the NLTK (Natural Language Toolkit) library. It is widely used for quick and easy sentiment analysis tasks, especially in situations where training a machine learning model for sentiment analysis may not be feasible or necessary. However, it's important to note that while VADER is a useful tool, it may not perform as well as more sophisticated machine learning models on certain types of data or in specific domains.

### Transformer Model

After a quick review of the scientific literature, we can learn that the state-of-the-art models for sentiment analysis are those based on a transform architecture such as the BERT model. Therefore, we propose to use a pre-trained then fine-tuned BERT model on a corpus of financial tweets to improve the computation of the sentiment score associated with each ECB statement.To do so we pre-trained model from `HuggingFace` library.

This model is based BERT (Bidirectional Encoder Representations from Transformers) is a pivotal model in the realm of natural language processing (NLP), and its innovation can be attributed to several key components, with the attention mechanism being a crucial aspect. Let's delve into its significance and the innovations behind it:

1. Pre-training and Fine-tuning:
Innovation: BERT introduced a two-step training process that leverages vast amounts of text data. First, the model is pre-trained on a large corpus of text using two unsupervised tasks: Masked Language Model (MLM) and Next Sentence Prediction (NSP). For example for the first task the model is trained to predict a word in the sentence given the context and meaning of the sentence .After pre-training, the model can be fine-tuned on specific downstream tasks, such as sentiment analysis or question-answering, with smaller, task-specific datasets. In our case the model we use is fine-tuned for the sentiement analysis task on a financial new dataset.

2. Bidirectional Context:
Innovation: Unlike previous models that processed text in a left-to-right or right-to-left manner, BERT utilizes a bidirectional approach. This means it considers the context from both directions (before and after a word) when encoding a word's representation. This bidirectional context helps in capturing a deeper understanding of words in their specific contexts.

3. Transformer Architecture:
Innovation: BERT is built upon the Transformer architecture, which was introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017). The Transformer's core is the attention mechanism, which allows it to weigh the significance of different words in a sentence when processing each word. This attention mechanism replaces the recurrent layers used in previous models and offers more parallelizable computations, making it more efficient and effective for capturing long-range dependencies in text.

#### Attention Mechanism in BERT

* **Self-Attention:** The attention mechanism in BERT allows the model to weigh the importance of different words (tokens) in a sequence when processing a particular word. This is termed as "self-attention" because it determines how much focus (attention weight) each word should give to every other word in the sequence.

* **Capturing Dependencies:** By employing self-attention, BERT can capture dependencies between words that are far apart in a sentence. For instance, in the sentence "The cat sat on the mat," understanding the relationship between "sat" and "mat" requires considering all the words in between, which self-attention can efficiently capture.

* **Multiple Layers and Heads:** BERT uses multiple attention layers (stacked on top of each other) and multiple attention heads within each layer. This design enables the model to capture various types of relationships in the words in the sentence, enhancing its ability to understand and represent complex linguistic structures and nuances.

In summary, the innovation behind BERT, especially its attention mechanism, revolutionized the field of NLP by enabling models to capture deeper contextual information from text data. By employing bidirectional context and the Transformer architecture's efficiency, BERT set new benchmarks in various NLP tasks, leading to significant advancements and applications in areas like machine translation, question-answering, sentiment analysis, and more.

## Models Evaluation

To discuss the result from a data science point of view we need to compare the prediction performance of the models against the ground-truth (i.e label of the dataset).

To do so, as we are in a multi-class classification task (0=bearish, 1=bullish, 0=neutral) we will use the confusion matrix to assess the performance of our models and select the best model.

### Naive Models

### VADER Algorithm

### BERT Models

## Results

| Model  | Accuracy          | Precision          | Recall          | F1-Score          |
| :--------------- |:---------------:|:---------------:|:---------------:|:---------------:|
| Naive Model {SentiWordNet}  |   0.42       |   0.37        |   0.38        |   0.36        |
| Naive Model {SentiWordNet + Negation Words + Booster Words} | 0.43             |   0.37       |   0.38        |   0.36        |
| Naive Model {SentiWordNet + Negation & Booster Words + Emoji}  | 0.42          |   0.37        |   0.38        |   0.36      |
| Vader Algorithm  | blabla          |   blabla        |   blabla       |  blabla       |
| ML Model {CountVectorizer, TF-IDF, MultinomialNB}  | 0.64          |   0.55        |   0.60       |  0.57       |
| BERT fine-tuned  | 0.88          |   0.84        |   0.85        |   0.85        |
| FinTweetBERT  | 0.95          |   0.92        |   0.97        |   0.94        |
| DistilRoBERTa  | 0.75          |   0.68        |   0.75        |   0.71        |
| FinBERT  | 0.73          |   0.65        |   0.70        |   0.67        |
