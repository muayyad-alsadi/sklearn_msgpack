# Scikit Learn Model Persistence via MsgPack 

## Introduction

Scikit learn [suggests](http://scikit-learn.org/stable/modules/model_persistence.html)
using [Pickle](https://docs.python.org/2/library/pickle.html)
to store model after training

There are known [issues](http://pyvideo.org/video/2566/pickles-are-for-delis-not-software) with this approach

* security - pickle contains byte codes
* maintainability - require same version of `sklearn`
* slow - because it contains byte codes not only trained weights

## Our approach

To persist a model instance, we construct a dictionary containing

* keyword params used to construct the instance
* the value of trainable parameters (ex. weights)
* other needed instance properties

The we use [MsgPack](https://msgpack.org/) to store this dictionary

## Supported classes

* [MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
* [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [TfidfTransformer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)

it's very easy to add more

## Performance

