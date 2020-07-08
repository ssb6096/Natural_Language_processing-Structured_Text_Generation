


#Doesnt Do Anything


import fastai
from fastai.text import *
from fastai.conv_learner import *
from fastai.column_data import *

PATH='/kaggle/working/nietzsche/'
get_data("https://s3.amazonaws.com/text-datasets/nietzsche.txt", f'{PATH}nietzsche.txt')
text = open(f'{PATH}nietzsche.txt').read()
print('corpus length:', len(text))