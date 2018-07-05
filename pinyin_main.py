import os, sys
import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *

itokens, otokens = dd.MakeS2SDict('data/pinyin.corpus.txt', dict_file='data/pinyin_word.txt')

print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())

from transformer import Transformer, LRSchedulerPerStep

d_model = 256  
n_head = 4
s2s = Transformer(itokens, otokens, len_limit=500, d_model=d_model, d_inner_hid=1024, \
				   n_head=n_head, d_k=d_model//n_head, d_v=d_model//n_head, layers=3, dropout=0.1)

mfile = 'models/pinyin.model.h5'
lr_scheduler = LRSchedulerPerStep(d_model, 4000) 
model_saver = ModelCheckpoint(mfile, monitor='ppl', save_best_only=True, save_weights_only=True)

#s2s.model.summary()
opt = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
s2s.compile(opt)

try: s2s.model.load_weights(mfile)
except: print('\n\nnew model')

cmds = sys.argv[1:]
if 'train' in cmds:
	gen = dd.S2SDataGenerator('data/pinyin.corpus.txt', itokens, otokens, batch_size=32, max_len=120)
	rr = next(gen); print(rr[0][0].shape, rr[0][1].shape)
	rr = next(gen); print(rr[0][0].shape, rr[0][1].shape)
	s2s.compile(opt, active_layers=1)
	s2s.model.fit_generator(gen, steps_per_epoch=2000, epochs=5, callbacks=[lr_scheduler, model_saver])
	s2s.compile(opt, active_layers=2)
	s2s.model.fit_generator(gen, steps_per_epoch=2000, epochs=5, callbacks=[lr_scheduler, model_saver])
	s2s.compile(opt, active_layers=3)
	s2s.model.fit_generator(gen, steps_per_epoch=2000, epochs=60, callbacks=[lr_scheduler, model_saver])
elif 'test' in cmds:
	print(s2s.decode_sequence_fast('ji zhi hu die zai yang guang xia fei wu 。'.split()))
	while True:
		quest = input('> ')
		print(s2s.decode_sequence_fast(quest.split()))
		rets = s2s.beam_search(quest.split())
		for x, y in rets: print(x, y)



