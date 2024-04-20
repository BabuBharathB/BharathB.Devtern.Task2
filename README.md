


# Spam Detection Using TensorFlow in Python

#Importing necessary libraries for EDA

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string

import nltk

from nltk.corpus import stopwords

from wordcloud import WordCloud

nltk.download('stopwords')

#Importing libraries necessary for Model Building and Training

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('spam_ham_dataset.csv')

data.head()

![Screenshot 2024-04-20 141541](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/2a62823a-7c10-43db-9707-17446689332d)



data.shape

![Screenshot 2024-04-20 141610](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/22649af6-a5fd-4d37-8a28-835f60f8f5b5)


sns.countplot( x='label_num',data=data)

plt.show()

![Screenshot 2024-04-20 141650](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/ad8a2016-b25a-46f6-bc68-0708948591c2)


#Downsampling to balance the dataset

ham_msg = data[data.label_num == 0]

spam_msg = data[data.label_num == 1]

ham_msg = ham_msg.sample(n=len(spam_msg),

                         random_state=42)
                         
#Plotting the counts of down sampled dataset

balanced_data = pd.concat([ham_msg, spam_msg])

balanced_data = balanced_data.reset_index(drop=True)

data.head()

plt.figure(figsize=(8, 6))

sns.countplot(data = balanced_data, x='label_num')

plt.title('Distribution of Ham and Spam email messages after downsampling')

plt.xlabel('Message types') 

![Screenshot 2024-04-20 141719](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/9b1664e5-552b-4443-a19d-a3c216cc13ea)



def remove_stopwords(text):

    stop_words = stopwords.words('english')
    
 
    imp_words = []
 
    # Storing the important words
    
    for word in str(text).split():
    
        word = word.lower()
        
 
        if word not in stop_words:
        
            imp_words.append(word)
            
 
    output = " ".join(imp_words)
 
    return output
    
#print(stopwords.words('english'))

balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text))

balanced_data.head(20)

![Screenshot 2024-04-20 141754](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/7727c33c-d0d9-4349-acbf-135e013cae41)




def plot_word_cloud(data, typ):

    email_corpus = " ".join(data['text'])
 
    plt.figure(figsize=(7, 7))
 
    wc = WordCloud(background_color='black',
    
                   max_words=100,
                   
                   width=800,
                   
                   height=400,
                   
                   collocations=False).generate(email_corpus)
 
    plt.imshow(wc, interpolation='bilinear')
    
    plt.title(f'WordCloud for {typ} emails', fontsize=15)
    
    plt.axis('off')
    
    plt.show()
 
plot_word_cloud(balanced_data[balanced_data['label_num'] == 0], typ='Non-Spam')

plot_word_cloud(balanced_data[balanced_data['label_num'] == 1], typ='Spam')


![Screenshot 2024-04-20 141825](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/a2b0a534-69e0-430a-9db9-ed296c304e31)



train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'],

                                                    balanced_data['label_num'],
                                                    
                                                    test_size = 0.2,
                                                    
                                                    random_state = 42)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_X)

 
#Convert text to sequences

train_sequences = tokenizer.texts_to_sequences(train_X)

test_sequences = tokenizer.texts_to_sequences(test_X)
 
#Pad sequences to have the same length

max_len = 100  # maximum sequence length

train_sequences = pad_sequences(train_sequences,

                                maxlen=max_len, 
                                
                                padding='post', 
                                
                                truncating='post')
                                
test_sequences = pad_sequences(test_sequences,

                               maxlen=max_len,
                               
                               padding='post',
                               
                               truncating='post')

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,

                                    output_dim=32, 
                                    
                                    input_length=max_len))
                                    
model.add(tf.keras.layers.LSTM(16))

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
 
#Print the model summary

model.summary()

![Screenshot 2024-04-20 141850](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/83483543-1726-4265-a0cb-bb677e25ec62)



model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),

              metrics = ['accuracy'],
              
              optimizer = 'adam')

from keras.callbacks import ReduceLROnPlateau

es = EarlyStopping(patience=3,

                   monitor = 'val_accuracy',
                   
                   restore_best_weights = True)
 
lr = ReduceLROnPlateau(patience = 2,

                       monitor = 'val_loss',
                       
                       factor = 0.5,
                       
                       verbose = 0)

history = model.fit(train_sequences, train_Y,

                    validation_data=(test_sequences, test_Y),
                    
                    epochs=20, 
                    
                    batch_size=32,
                    
                    callbacks = [lr, es]
                    
                   )

![Screenshot 2024-04-20 141912](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/49b688f9-58ef-4d0a-937a-8e6600d6e070)



test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)

print('Test Loss :',test_loss)

print('Test Accuracy :',test_accuracy)

![Screenshot 2024-04-20 141936](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/4ee2b2f1-7638-46c3-968d-9ce2f701ce97)




test_X.head()

![Screenshot 2024-04-20 142003](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/e8774953-430d-46fc-915b-71e43a124ce1)


plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend()

plt.show()


![Screenshot 2024-04-20 142037](https://github.com/BabuBharathB/BharathB.Devtern/assets/167573509/2046efd2-1d04-4af5-bca9-30547bcfdb2c)


