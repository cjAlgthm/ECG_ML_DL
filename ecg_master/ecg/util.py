import os
import pickle #import cPickle as pickle  cj modify

def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as fid: #r->rb cj modify
        preproc = pickle.load(fid)
    return preproc

def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f,'wb') as fid:   #w->wb cj modify
        pickle.dump(preproc,fid,0) # add 0 cj modify
