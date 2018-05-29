# ModelJack

Author: Nerses Nersesyan

## Abstract


This project shows how to work with the various data sets in Wikipedia Talk project on Figshare using [fasttext](https://fasttext.cc/). Wikipedia Talk project release includes:
1. large historical [corpus](https://figshare.com/articles/Wikipedia_Talk_Corpus/4264973) of discussion comments on Wikipedia talk pages
2. [sample](https://figshare.com/articles/Wikipedia_Detox_Data/4054689) of over 100k comments with human labels for whether the comment contains a personal attack
3. [sample](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) of over 100k comments with human labels for whether the comment has aggressive tone

Please refer to [wiki](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release) for documentation of the schema of each data set and the [research paper](https://arxiv.org/abs/1610.08914) for documentation on the data collection and modeling methodology.

It is important to note that there is an excisting [API](https://www.perspectiveapi.com/#/) demo version created by [Jigsaw](https://jigsaw.google.com/). The API scores a comment based on its potential impact on a conversation.More detailed information about this project can be found [here](https://conversationai.github.io/).

In this notebook we show how to build a simple classifier using [fasttext](https://fasttext.cc/) for detecting personal attacks and apply the classifier to a random sample of the comment corpus to see whether discussions on user pages have more personal attacks than discussion on article pages.
