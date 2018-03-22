import os, sys
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *

itokens, otokens = dd.MakeS2SDict('en2de.s2s.txt', dict_file='en2de_word.txt')
Xtrain, Ytrain = dd.MakeS2SData('en2de.s2s.txt', itokens, otokens, h5_file='en2de.h5')
Xvalid, Yvalid = dd.MakeS2SData('en2de.s2s.valid.txt', itokens, otokens, h5_file='en2de.valid.h5')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

'''
from rnn_s2s import RNNSeq2Seq
s2s = RNNSeq2Seq(itokens,otokens, 256)
s2s.compile('rmsprop')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, validation_data=([Xvalid, Yvalid], None))
'''

from transformer import Transformer, LRSchedulerPerStep, LRSchedulerPerEpoch

d_model = 256
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
				   n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1)

lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
#lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint('en2de.model.h5', save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()
try: s2s.model.load_weights('en2de.model.h5')
except: print('\n\nnew model')
s2s.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, \
				validation_data=([Xvalid, Yvalid], None), \
				callbacks=[lr_scheduler, model_saver])
