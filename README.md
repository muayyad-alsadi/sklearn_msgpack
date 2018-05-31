# Scikit Learn Model Persistence via MsgPack 

## Introduction

Scikit learn [suggests](http://scikit-learn.org/stable/modules/model_persistence.html)
using [Pickle](https://docs.python.org/2/library/pickle.html)
to store model after training

There are known [issues](http://pyvideo.org/video/2566/pickles-are-for-delis-not-software) with this approach

* security - pickle contains byte codes
* maintainability - require same version of `sklearn` 
* slow - because it contains byte codes not only trained weights

```
UserWarning: Trying to unpickle estimator MLPClassifier from version 0.18 when using version 0.19.1. This might lead to breaking code or invalid results. Use at your own risk.
```

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

We have seen more than 25x faster loading for `MLPClassifier`
and 150x faster loading for `TfidfVectorizer`

And in terms of size, 7x smaller files for `MLPClassifier` and 50x smaller for `TfidfVectorizer`

## Usage


```
import sklearn_msgpack
sklearn_msgpack.save_to_file('tmp.mpack', clf)
# ...
clf = sklearn_msgpack.load_from_file('tmp.mpack')
```

