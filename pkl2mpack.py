#! /usr/bin/env python

import os
import sys

import cPickle as pickle

from scikit_msgpack import generic_save

HERE = os.path.dirname(__file__)
input_fn = os.path.join(HERE, sys.argv[1])
output_fn = input_fn.replace('.pkl', '.mpack')
if not input_fn.endswith('.pkl'):
    raise ValueError('input file does not end in .pkl')
with open(input_fn, 'r') as f:
    instance = pickle.load(f)
with open(output_fn, 'wb') as f:
    generic_save(instance, f)
