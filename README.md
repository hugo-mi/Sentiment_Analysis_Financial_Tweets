# Sentiment_Analysis_Financial_Tweets

## DATA

The data we use comes from HuggingFace "zeroshot/twitter-financial-news-sentiment"
The dataset holds 11,932 documents annotated with 3 labels:
* Bearish = 0
* Bullish = 1
* Neutral = 2

## Goals

Assess the prediction performance on sentiment classification task on financial tweets

### Models

#### Naive model

We will apply a naïve model to predict the sentiment score of set of tweets, To do so, we will use a dictionary `SentiWordNet`
We will improve our model by taking into account of potential emoji and negation and booster words in the tweets.

#### Vader Algorithm

VADER Model is a specific model for tweet analysis and can take into account emoji

#### Transformer Model



#### Model Evaluation

To discuss the result from a data science point of view we need to compare the prediction performance of the models against the ground-truth (i.e label of the dataset).

To do so, as we are in a multi-class classification task (0=bearish, 1=bullish, 0=neutral) we will use the confusion matrix to assess the performance of our models and select the best model.

**Naive Model**

## Results


