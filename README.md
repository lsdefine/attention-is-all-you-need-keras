# The Transformer model in Attention is all you need：a Keras implementation.
A Keras+TensorFlow Implementation of the Transformer: "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017)

# Usage
Please refer to *en2de_main.py* and *pinyin_main.py*
### en2de_main.py
- This task is same as in [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch): WMT'16 Multimodal Translation: Multi30k (de-en) [(http://www.statmt.org/wmt16/multimodal-task.html)](http://www.statmt.org/wmt16/multimodal-task.html). We borrowed the data proprocessing step 0 and 1 in the repository, and then construct the input file *en2de.s2s.txt*
#### Results
- The code achieves near results as in the repository: about 70% valid accuracy. 
If using smaller model parameters, such as *layers=2* and *d_model=256*, the valid accuracy is better since the task is quite small.
### For your own data
- Just preproess your source and target sequences as the format in *en2de.s2s.txt* and *pinyin.corpus.examples.txt*.
### Some notes
- For larger number of layers, the special learning rate scheduler reported in the papar is necessary.
- In *pinyin_main.py*, I tried another method to train the deep network. I train the first layer and the embedding layer first, then train a 2-layers model, and then train a 3-layers, etc. It works in this task.

# Acknowledgement
- Some model structures and some scripts are borrowed from [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).
