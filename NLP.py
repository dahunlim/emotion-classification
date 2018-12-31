import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from konlpy.tag import Twitter

%matplotlib inline

#Data Read
inputData = open("data_2520.txt", 'rt', encoding='UTF8')
text = inputData.read()
x_train = []
y_train = []
lines = text.split("\n")
for line in lines:
    x_train.append(line)

inputLabel = open("label_2520.txt", 'rt', encoding='UTF8')
textL = inputLabel.read()


labels = textL.split("\n")
for line in labels:
	y_train.append(line)


#Data Preprocessing
twitter = Twitter()

x_train_pos = []
i = 0

for sentence in x_train:
    sentence_cleaned = ''
    sentence_pos = twitter.pos(sentence, norm=True, stem=True)
    for item in sentence_pos:
#         if (item[1] != ['josa', 'punctuation', 'eomi', 'number']):
        if item[1] in ['Noun', 'Adjective', 'Determiner','Exclamation', 'Unknown', 'KoreanParticle']:
            #print('%s : %s' % (item[0], item[1]))
            sentence_cleaned += item[0] + ' '
    x_train_pos.append(sentence_cleaned)
    print(i)
    i += 1

#Vectorizer
vect = CountVectorizer()
vect.fit(x_train_pos)
result = vect.transform(x_train_pos).toarray()

#TF-IDF
vect = TfidfTransformer()
vect.fit(result)
result = vect.transform(result).toarray()

#Data Split
X_train,X_test,Y_train,Y_test = train_test_split(result,y_train,test_size=0.1)

Y_train = np.array(Y_train)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

max_words = 1500
max_len = 120
sequences_matrix = sequence.pad_sequences(X_train,maxlen=max_len)

#LSTM
def RNN():
   inputs = Input(name='inputs', shape=[max_len])
   layer = Embedding(max_words, 50, input_length=max_len)(inputs)
   layer = LSTM(64)(layer)
   layer = Dense(256,name='FC1')(layer)
   layer = Activation('relu')(layer)
   layer = Dropout(0.5)(layer)
   layer = Dense(6,name='out_layer')(layer)
   layer = Activation('sigmoid')(layer)
   model = Model(inputs=inputs,outputs=layer)
   return model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
          validation_split=0.2)

test_sequences_matrix = sequence.pad_sequences(X_test,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))