import numpy as np
import scipy.sparse as sp

import msgpack

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_random_state

types_list=[
  np.int32, np.int64,
  np.uint32, np.uint64,
  np.float32, np.float64,
]
types_map=dict([( type(t()).__name__, t ) for t in types_list ])

def type_from_string(v):
    given_type = types_map.get(v)
    if not given_type: raise ValueError('Unknown type [%r]' % v)
    return type(given_type())

types_map['type']=type_from_string
types_map['ndarray']=np.array
types_map['csr_matrix'] = lambda kw: sp.csr_matrix(
  tuple(kw['arg1']),
  shape=tuple(kw['shape']),
  dtype=type_from_string(kw['dtype']),
)
types_map['tuple']=tuple

simplify_type = {
    'int32': lambda i: int(i),
    'int64': lambda i: int(i),
    'uint32': lambda i: int(i),
    'uint32': lambda i: int(i),
    'float32': lambda i: float(i),
    'float64': lambda i: float(i),
    'ndarray': lambda i: i.tolist(),
    'tuple': lambda i: list(i),
    'csr_matrix': lambda i: {
        'shape': list(i.shape),
        'arg1': [i.data.tolist() , i.indices.tolist(), i.indptr.tolist()], 
        'dtype': i.dtype.name,
    },
}

def get_value_n_type(value):
    type_name = type(value).__name__
    if type_name == 'type':
        return value.__name__, 'type'
    if type_name in simplify_type:
        return simplify_type[type_name](value), type_name
    return value, None

def value_to_type(value, type_name=None):
    if type_name is None: return value
    if type_name not in types_map: return value
    return types_map[type_name](value)

def get_simplified_params(params, types):
    ret={}
    for key, value in params.items():
        simplified, type_name = get_value_n_type(value)
        if type_name is not None: types[key] = type_name
        ret[key] = simplified
    return ret

def load_simplified_params(dst, params, types):
    for key, value in params.items():
        type_name = types.get(key)
        value = value_to_type(value, type_name)
        print "setting ", key
        setattr(dst, key, value)


def load_x_y_csv(fd):
    X=[]
    Y=[]
    for line in fd.readlines():
        parts=line.strip().split(',')
        y=int(parts[0])
        Y.append(y)
        x=[ float(part) for part in parts[1:] ]
        X.append(x)
    X=np.array(X)
    Y=np.array(Y)
    return X, Y

def forward_pass(clf, X, to=None):
    if to is None:
        to = clf.n_layers_ - 1
    activations = [X]
    activations.extend([None for i in range(1, clf.n_layers_ - 1) ])
    hidden_activation = ACTIVATIONS[clf.activation]
    # Iterate over the hidden layers
    for i in range(to):
        activations[i + 1] = safe_sparse_dot(activations[i], clf.coefs_[i])
        activations[i + 1] += clf.intercepts_[i]

        # For the hidden layers
        if (i + 1) != (clf.n_layers_ - 1):
            activations[i + 1] = hidden_activation(activations[i + 1])
        else:
            output_activation = ACTIVATIONS[clf.out_activation_]
            activations[i + 1] = output_activation(activations[i + 1])

    return activations

def mlp_get_params(src, keep_loss=True, keep_iter=True, keep_weights=True):
    ret={"class_name": src.__class__.__name__}
    params = src.get_params(True)
    ret['params']=params
    xtra={}
    types={}
    attrs=['classes_', 'n_layers_', 'n_outputs_', 'loss', '_estimator_type', 'out_activation_']
    for attr in attrs:
        value=getattr(src, attr, None)
        xtra[attr]=value
        types[attr]=type(value).__name__
    if keep_weights:
        weights={}
        for attr in ['intercepts_', 'coefs_']:
            weights[attr]=getattr(src, attr)
        ret['weights'] = weights
    if keep_loss:
        for attr in ['loss_', 'best_loss_', 'loss_curve_']:
            value=getattr(src, attr, None)
            if value is not None: xtra[attr]=value
    if keep_iter:
        for attr in ['t_', 'n_iter_', '_no_improvement_count']:
            value=getattr(src, attr, None)
            if value is not None: xtra[attr]=value
    ret['xtra'] = xtra
    ret['types'] = types
    #if src._label_binarizer.classes_!=src.classes_:
    ret['label_binarizer_classes_']=list(src._label_binarizer.classes_)
    return ret

def mlp_load_params(dst, mlp_params, **kw):
    params=mlp_params['params']
    dst.set_params(**params)
    xtra=mlp_params.get('xtra', {})
    types=mlp_params.get('types', {})
    for key, value in xtra.items():
        value_type = types.get(key, None)
        if (value_type=='tuple'): value=tuple(value)
        setattr(dst, key, value)
    if 'weights' in mlp_params:
        weights = mlp_params['weights']
        dst.coefs_ = [ np.array(a) for a in weights['coefs_'] ]
        dst.intercepts_ = np.array(weights['intercepts_'])
    for key, value in kw.items():
        setattr(dst, key, value)
    if not hasattr(dst, 'loss_'): dst.loss_=1e6
    if not hasattr(dst, 'best_loss_'): dst.best_loss_=dst.loss_
    if not hasattr(dst, 'loss_curve_'): dst.loss_curve_=[dst.best_loss_]
    if not hasattr(dst, 't_'): dst.t_=0
    if not hasattr(dst, 'n_iter_'): dst.n_iter_=0
    if not hasattr(dst, '_no_improvement_count'): dst._no_improvement_count=0
    if not hasattr(dst, '_label_binarizer'):
        classes = mlp_params.get('label_binarizer_classes_', xtra.get('classes_', None))
        dst._label_binarizer = LabelBinarizer()
        dst._label_binarizer.fit(classes)
        dst._label_binarizer.classes_=classes
    if hasattr(dst, '_optimizer'): del dst._optimizer
    return dst

def mlp_construct(mlp_params, **kw):
    dst = MLPClassifier()
    mlp_load_params(dst, mlp_params, **kw)
    return dst

# TODO: consider array
# from array import array
# W=np.array(...)
# original_shape = W.shape
# a=array('d', W.reshape(-1)) # then tofile() or tostring()
# read it again
# np.array(  ).reshape( original_shape )


def mlp_save(mlp, fd, **kw):
    msgpack.dump(mlp_get_params(mlp, **kw), fd, default=lambda o: list(o))

def mlp_load(fd, **kw):
    return mlp_construct(msgpack.load(fd), **kw)

def tfidf_transformer_get_params(tfidf):
    types={}
    params = get_simplified_params(tfidf.get_params(), types)
    xtra_names=['_idf_diag']
    xtra = dict([(key, getattr(tfidf, key)) for key in xtra_names])
    xtra = get_simplified_params(xtra, types)
    return {
      'class_name': tfidf.__class__.__name__,
      'params': params,
      'xtra': xtra,
      'types': types,
    }

def tfidf_transformer_load_params(dst, params, **kw):
    load_simplified_params(dst, params['params'], params['types'])
    load_simplified_params(dst, params['xtra'], params['types'])
    for key, value in kw.items():
        setattr(dst, key, value)
    return dst

def tfidf_transformer_construct(params, **kw):
    dst = TfidfTransformer()
    tfidf_transformer_load_params(dst, params, **kw)
    return dst


def tfidf_vectorizer_get_params(vectorizer, keep_stop_words=False):
    types={}
    params = get_simplified_params(vectorizer.get_params(), types)
    xtra_names=['vocabulary_']
    if keep_stop_words:
        xtra_names.push('stop_words_')
    xtra = dict([(key, getattr(vectorizer, key)) for key in xtra_names])
    xtra = get_simplified_params(xtra, types)
    
    ret = {
      'class_name': vectorizer.__class__.__name__,
      'params': params,
      'xtra': xtra,
      'types': types,
      'transformer': tfidf_transformer_get_params(vectorizer._tfidf),
    }
    return ret

def tfidf_vectorizer_load_params(dst, params, **kw):
    load_simplified_params(dst, params['params'], params['types'])
    load_simplified_params(dst, params['xtra'], params['types'])
    dst._tfidf = tfidf_transformer_construct(params['transformer'])
    for key, value in kw.items():
        setattr(dst, key, value)
    return dst

def tfidf_vectorizer_construct(params, **kw):
    dst = TfidfVectorizer()
    tfidf_vectorizer_load_params(dst, params, **kw)
    return dst

def tfidf_vectorizer_save(vectorizer, fd, **kw):
    msgpack.dump(tfidf_vectorizer_get_params(vectorizer, **kw), fd, default=lambda o: list(o))

def tfidf_vectorizer_load(fd, **kw):
    return tfidf_vectorizer_construct(msgpack.load(fd, encoding='utf-8'), **kw)

def generic_save(instance, fd, **kw):
    if isinstance(instance, MLPClassifier):
        mlp_save(instance, fd, **kw)
    elif isinstance(instance, TfidfVectorizer):
        tfidf_vectorizer_save(instance, fd, **kw)
    else:
        raise TypeError('only accepts MLPClassifier or TfidfVectorizer')

def generic_load(fd, **kw):
    loaded = msgpack.load(fd, encoding='utf-8')
    class_name = loaded.get('class_name')
    if class_name == 'MLPClassifier':
        print "loading MLPClassifier"
        return mlp_construct(loaded, **kw)
    elif class_name == 'TfidfVectorizer':
        print "loading TfidfVectorizer"
        return tfidf_vectorizer_construct(loaded, **kw)
    else:
        raise TypeError('only accepts MLPClassifier or TfidfVectorizer')
