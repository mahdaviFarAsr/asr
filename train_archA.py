from network_archA import ARCH_A
import pandas as pd
import pickle

with open('PersianDict.pickle', 'rb') as handle:
    PersianDict = pickle.load(handle)

with open('NumberDict.pickle', 'rb') as handle:
    NumberDict = pickle.load(handle)

train = pd.read_csv('train_archA.csv')
train_wavefile = train.values.tolist()
basic_path = 'H:/asr/sorceCode/code/software/ParsSeda/audio/'
export_path = 'mymodel/'

model_archA = ARCH_A(basic_path, train_wavefile, PersianDict, NumberDict, export_path)

model_archA.train()
