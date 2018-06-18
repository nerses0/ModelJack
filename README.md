# ModelJack

Author: Nerses Nersesyan

ModelJack is project for effectively emulating interesting language APIs with simple models that can run locally to avoid latency, costs and request limits.

### Example 1: Comment Toxicity

In the first example we show how to train a model that emulates the Google Perspective API using data from the Wikipedia Talk project and the [fasttext](https://fasttext.cc/) library.

The [Perspective API](https://www.perspectiveapi.com/#/) is a demo released by the Google [Jigsaw](https://jigsaw.google.com/) team. The API scores a comment based on its potential impact on a conversation, deting personal attacks.  More detailed information about the project can be found [here](https://conversationai.github.io/).

Detecting and reducing toxic comments and personal attacks is very important for most platforms with user-generated content.  The Perspective API is potentially very useful, but is a demo limited to 1000 requests.

Can we emulate it so that developers can integrate this functionality into their platforms today?

#### Running

Download the data to this directory and run:
```
python ft_cls.py
```

#### Results

See [Results](Results.md)


#### Datasets

For training and evaluation of created model we used the Wikipedia Talk project dataset.

Wikipedia Talk project release includes:  

1. a large historical [corpus](https://figshare.com/articles/Wikipedia_Talk_Corpus/4264973) of discussion comments on Wikipedia talk pages  

2. a [sample](https://figshare.com/articles/Wikipedia_Detox_Data/4054689) of over 100k comments with human labels for whether the comment contains a personal attack

3. a [sample](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973) of over 100k comments with human labels for whether the comment has an aggressive tone

Please refer to [meta.wikimedia.org/wiki/Research:Detox/Data_Release](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release) for documentation of the schema of each data set.


## References
[*Ex Machina: Personal Attacks Seen at Scale*](https://arxiv.org/abs/1610.08914) - documentation on the data collection and modeling methodology from Google and Wikimedia

[Conversation AI ](https://github.com/conversationai) - The Conversation AI Research Github Organization at Google
