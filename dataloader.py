import os, sys, time, random
import ljqpy
import h5py
import numpy as np

class TokenList:
	def __init__(self, token_list):
		self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
		self.t2id = {v:k for k,v in enumerate(self.id2t)}
	def id(self, x):	return self.t2id.get(x, 1)
	def token(self, x):	return self.id2t[x]
	def num(self):		return len(self.id2t)
	def startid(self):  return 2
	def endid(self):    return 3
	
def pad_to_longest(xs, tokens, max_len=999):
	longest = min(len(max(xs, key=len))+2, max_len)
	X = np.zeros((len(xs), longest), dtype='int32')
	X[:,0] = tokens.startid()
	for i, x in enumerate(xs):
		x = x[:max_len-2]
		for j, z in enumerate(x):
			X[i,1+j] = tokens.id(z)
		X[i,1+len(x)] = tokens.endid()
	return X

def MakeS2SDict(fn=None, min_freq=5, delimiter=' ', dict_file=None):
	if dict_file is not None and os.path.exists(dict_file):
		print('loading', dict_file)
		lst = ljqpy.LoadList(dict_file)
		midpos = lst.index('<@@@>')
		itokens = TokenList(lst[:midpos])
		otokens = TokenList(lst[midpos+1:])
		return itokens, otokens
	data = ljqpy.LoadCSV(fn)
	wdicts = [{}, {}]
	for ss in data:
		for seq, wd in zip(ss, wdicts):
			for w in seq.split(delimiter): 
				wd[w] = wd.get(w, 0) + 1
	wlists = []
	for wd in wdicts:	
		wd = ljqpy.FreqDict2List(wd)
		wlist = [x for x,y in wd if y >= min_freq]
		wlists.append(wlist)
	print('seq 1 words:', len(wlists[0]))
	print('seq 2 words:', len(wlists[1]))
	itokens = TokenList(wlists[0])
	otokens = TokenList(wlists[1])
	if dict_file is not None:
		ljqpy.SaveList(wlists[0]+['<@@@>']+wlists[1], dict_file)
	return itokens, otokens

def MakeS2SData(fn=None, itokens=None, otokens=None, delimiter=' ', h5_file=None, max_len=200):
	if h5_file is not None and os.path.exists(h5_file):
		print('loading', h5_file)
		with h5py.File(h5_file) as dfile:
			X, Y = dfile['X'][:], dfile['Y'][:]
		return X, Y
	data = ljqpy.LoadCSVg(fn)
	Xs = [[], []]
	for ss in data:
		for seq, xs in zip(ss, Xs):
			xs.append(list(seq.split(delimiter)))
	X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
	if h5_file is not None:
		with h5py.File(h5_file, 'w') as dfile:
			dfile.create_dataset('X', data=X)
			dfile.create_dataset('Y', data=Y)
	return X, Y

def S2SDataGenerator(fn, itokens, otokens, batch_size=64, delimiter=' ', max_len=999):
	Xs = [[], []]
	while True:
		for ss in ljqpy.LoadCSVg(fn):
			for seq, xs in zip(ss, Xs):
				xs.append(list(seq.split(delimiter)))
			if len(Xs[0]) >= batch_size:
				X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
				yield [X, Y], None
				Xs = [[], []]

if __name__ == '__main__':
	itokens, otokens = MakeS2SDict('en2de.s2s.txt', 6, dict_file='en2de_word.txt')
	X, Y = MakeS2SData('en2de.s2s.txt', itokens, otokens, h5_file='en2de.h5')
	print(X.shape, Y.shape)
