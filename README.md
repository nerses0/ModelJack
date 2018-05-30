# ModelJack

Author: Nerses Nersesyan

## Abstract


This project shows how to work with the various data sets in Wikipedia Talk project on Figshare using [fasttext](https://fasttext.cc/). 

It is important to note that there is an excisting [API](https://www.perspectiveapi.com/#/) demo version created by [Jigsaw](https://jigsaw.google.com/). The API scores a comment based on its potential impact on a conversation.More detailed information about this project can be found [here](https://conversationai.github.io/).

In this notebook we show how to build a simple classifier using [fasttext](https://fasttext.cc/) for detecting personal attacks and apply the classifier to a random sample of the comment corpus to see whether discussions on user pages have more personal attacks than discussion on article pages.

## Impact
Quantity of social media platforms users is rising from day to day and online discussion has become integral to people’s experience of the internet. It would be naive to have ever expected that online discussion won't contain abuse or harrasment. Manually moderating comments and discussion forums can be tedious and expensive. That's why any tool which is capable to increase moderation quality and decrease it's expenses would be in demand.

## Existing work
[Research paper](https://arxiv.org/abs/1610.08914) containing documentation on the data collection and modeling methodology.

## Roadmap
*Deliverable*

Create a classifier using fastext with accuracy higher than 90%.

*Milestone 1*

Building a classifier based on fasttext for personal attacks

*Milestone 2*
- Model tune
- Use of classifier on the [Wikipedia Talk Corpus](https://figshare.com/articles/Wikipedia_Talk_Corpus/4264973) 

## Resources

For training and evaluation of created model were used Wikipedia Talk project dataset. Wikipedia Talk project release includes:
1. large historical [corpus](https://figshare.com/articles/Wikipedia_Talk_Corpus/4264973) of discussion comments on Wikipedia talk pages
2. [sample](https://figshare.com/articles/Wikipedia_Detox_Data/4054689) of over 100k comments with human labels for whether the comment contains a personal attack
3. [sample](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) of over 100k comments with human labels for whether the comment has aggressive tone

Please refer to [wiki](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release) for documentation of the schema of each data set.
