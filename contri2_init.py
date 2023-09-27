import Nettoyage as net
import nltk
import pandas as pd
import numpy as np 
import cv2 as cv
#from fastai.imports import *
import os, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import timeit
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
tf.gfile = tf.io.gfile
import tensorflow_hub as hub
from tensorflow import keras
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from keras.layers import Bidirectional,LSTM, Add,GRU,MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout,Conv1D,Embedding,Flatten, Input, Layer,GlobalAveragePooling1D,Activation,Lambda,LayerNormalization, Concatenate, Average,AlphaDropout,Reshape, multiply

import contractions
#from bs4 import BeautifulSoup
from keras.utils import to_categorical
from sklearn import preprocessing
#from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer,PorterStemmer
from tensorflow.keras.layers import TextVectorization
import tqdm
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import roc_curve, auc
import spacy
from scipy import stats
from spacy import displacy
#nlp = spacy.load("en_core_web_sm")
#import bert_tokenizer as tok
#import absl.logging
import tensorflow_hub as hub
from bert import tokenization
#absl.logging.set_verbosity(absl.logging.ERROR)

################# DEBUT..................DEFINITION DES FONCTIONS ###################
best_models = []
m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def fake_virtual(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    #input1 = Input(shape=(100,))
    #embedding_layer = Embedding(len(word_index)+1,100,embeddings_initializer=keras.initializers.Constant(embedding_matrix),input_length=100,trainable=False)(input1)
    #model1 = Bidirectional(LSTM(32))(embedding_layer)
    model1 = Bidirectional(LSTM(32))(sequence_output)
    model1 = Dense(64, activation='softmax')(model1)

    input2 = Input(shape=(224,224,3))
    model2 = Conv2D(filters=3, kernel_size=5,kernel_constraint=CustomConstraint(3,5), padding='same', use_bias=False)(input2)
    model2 = Conv2D(filters=16,kernel_size=3, padding='same',use_bias=False)(model2)
    model2 = BatchNormalization(axis=3, scale=False)(model2)
    model2 = Activation('relu')(model2)
    model2 = Conv2D(filters=32,kernel_size=3, padding='same',use_bias=False)(model2)
    model2 = BatchNormalization(axis=3, scale=False)(model2)
    model2 = Activation('relu')(model2)
    model2 = GlobalAveragePooling2D()(model2)
    model2 = Dense(64, activation='relu')(model2)

    outFinal = tf.keras.layers.Add()([model1, model2])
    final_model_output = Dense(2, activation='softmax')(outFinal)
    #input1=[input_word_ids, input_mask, segment_ids]
    final_model = Model(inputs=[input_word_ids, input_mask, segment_ids, input2], outputs=final_model_output)
    #final_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy", f1_m])
    final_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
    return final_model

#K.clear_session()

class CustomConstraint(Constraint):
    n=5

    def __init__(self,k,s):
        self.k = k
        self.s = s
        #np.random.rand2 = lambda *args, dtype = np.float32: np.random.rand(*args).astype(dtype)
        f1 = np.random.rand(s,s).astype('f')
        custom_weights = np.array((f1, f1, f1))
        f2 = custom_weights.transpose(1, 2, 0)
        custom_weights = np.tile(f2, (k, 1, 1))
        T2 = np.reshape(custom_weights,(k,s,s,3))
        custom_weights = T2.transpose(1, 2, 3, 0)
        self.custom_weights = tf.Variable(custom_weights)
    def __call__(self, weights):
        weights = self.custom_weights
        row_index = self.s//2
        col_index = self.s//2
        new_value = 0
        weights[row_index,col_index,:,:].assign(new_value)
        som = tf.keras.backend.sum(weights)
        sum_without_center1 = 1/som
        newMatrix = weights*sum_without_center1
        weights.assign(newMatrix)
        new_value = -1
        weights[row_index,col_index,:,:].assign(new_value)
        return weights

def plot_roc_curve(fper, tper,fold_var):
    plt.plot(fper, tper, color = 'red', label = 'ROC')
    plt.plot([0, 1], [0, 1], color = 'green', linestyle = '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig('courbes_ROC_'+str(fold_var)+'.png')

def plot_loss(epochs, loss,val_loss):
  plt.plot(epochs, loss, 'b', label = 'loss')
  plt.plot(epochs, val_loss, 'r', label = 'Val loss')
  plt.title('Loss and Val_Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('courbes_loss.png')

def plot_accuracy(epochs, accuracy,val_accuracy):
  plt.plot(epochs, accuracy, 'b', label = 'Accuracy')
  plt.plot(epochs, val_accuracy, 'r', label = 'Val Accuracy')
  plt.title('Accuracy and Val Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig('courbes_Accuracy.png')

def predictModel(X):
    testText = bert_encode(X['text'])
    testImage = X['image']
    testImage = testImage.to_numpy()
    testImage = np.array([val for val in testImage])
    testLabel = label.fit_transform(X['label'])
    testLabel = to_categorical(testLabel)
    testLabel = testLabel
    y_pred = best_models[0].predict([testText,testImage])
    return y_pred

def plot_learning_curve(best_models):
    epochs = 30
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 1 * 6), dpi = 100)
    # Classification Report curve
    sns.lineplot(x = np.arange(1, epochs + 1), y = best_models[0].history.history['acc'],palette = ['b'], ax = axes[0][0],label = 'train_accuracy')
    sns.lineplot(x = np.arange(1, epochs + 1), y = best_models[0].history.history['val_acc'],palette = ['r'], ax = axes[0][0],label = 'val_accuracy')       
    axes[0][0].legend()
    # Loss curve
    sns.lineplot(x = np.arange(1, epochs + 1), y = best_models[0].history.history['loss'],palette = ['b'], ax = axes[0][1], label = 'train_loss')
    sns.lineplot(x = np.arange(1, epochs + 1), y = best_models[0].history.history['val_loss'],palette = ['r'], ax = axes[0][1], label = 'val_loss')
    axes[0][1].legend() 
    for j in range(2):
        axes[0][j].set_xlabel('Epoch', size=12)
    plt.savefig('courbes_Accuracy_loss.png')
    plt.show()
def confusionMatrix(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    class_names = ['fake','real']
    #Transform to df for easier plotting
    cm_df = pd.DataFrame(cm)
    final_cm = cm_df
    plt.figure(figsize = (5,5))
    sns.heatmap(final_cm, annot = True,cmap = "YlGnBu",cbar = False,fmt = 'd')
    plt.title('Confusion matrix', y = 1.1)
    plt.ylabel('Actual class')
    plt.xlabel('Prediction class')
    plt.savefig('ConfusionMatrice.png')
    plt.show()

def plotCurve(y_test,y_pred):
    #define metrics
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    #create ROC curve
    plt.plot(fpr,tpr,color='red',label = "AUC = "+str(auc))
    plt.plot([0, 1], [0, 1], color = 'green')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 4)
    plt.savefig('ROC_Curve.png')
    plt.show()
################# FIN DEFINTION DES FONCTIONS ##########

dfTrain = pd.read_csv('data_nettoyerTrain.csv')
dfTest = pd.read_csv('data_nettoyerTest.csv')

################ IMAGES INSERTION ################
dataImageDev = pd.DataFrame(columns = ['nomImage','image','label'])
dataImageTest = pd.DataFrame(columns = ['nomImage','image','label'])

textListe = []
imageListe = []
labelImage = []
labelText = []
#Former un dataframe les images et le label
i=0
j=0
repDev=glob.glob('MediaEval2016/DevImages/*')

while i < len(repDev):
  try:
    img2 = cv.imread(repDev[i])
    imgResize = cv.resize(img2, (224,224))
    chemin= repDev[i].split("/")
    imgName = chemin[len(chemin)-1]
    imgName = imgName.replace(" ", "")
    nb = len(imgName)-4
    dataImageDev.loc[i] = [imgName[:nb],imgResize,0]
  except:
    pass
  i=i+1

dataImageTextDev = pd.DataFrame(columns=['text','nomImage','image','label'])
dataImageTextTest = pd.DataFrame(columns=['text','nomImage','image','label'])
imageListe = []
labelImage = []
textListe = []
k=0
j=0
i=0
while i <len(dfTrain):
  #name = dataOtre.loc[i,'imageId(s)']
  name = dfTrain['imageId(s)'][i]
  trouver = 0
  for index, valeur in dataImageDev['nomImage'].items():
    if(name == valeur):
      trouver = 1
      tweetClean = dfTrain['tweetText'][i]
      dataImageTextDev.loc[k] = [str(tweetClean),name,dataImageDev['image'][index],dfTrain['label'][i]]
      k = k+1
  i = i+1

i = 0
while i <len(dfTest):
  name = dfTest['imageId(s)'][i]
  trouver=0
  for index, valeur in dataImageDev['nomImage'].items():
    if(name == valeur):
      trouver = 1
      tweetClean = dfTest['tweetText'][i]
      dataImageTextTest.loc[k] = [str(tweetClean),name,dataImageDev['image'][index],dfTest['label'][i]]
      k=k+1
  i=i+1

################ FIN IMAGES INSERTION ##############

############### RESUME DU MODEL  #############
max_len = 26
model = fake_virtual(bert_layer, max_len = max_len)
model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names = True)
############### FIN RESUME DU MODEL  #############

############# ENTRAINEMENT DU MODEL ##########
i=0
kf = KFold(n_splits = 5, shuffle = True)
save_dir = './saved_models/'
fold_var = 1
max_len = 26
result = []
scores_loss = []
scores_acc = []
scores_pre = []
scores_rap = []
fold_var = 0

for train_indices, val_indices in kf.split(dataImageTextDev):
    train = dataImageTextDev.iloc[train_indices]
    val = dataImageTextDev.iloc[val_indices]
    label = preprocessing.LabelEncoder()

    trainText = bert_encode(train['text'], tokenizer, max_len=max_len)
    trainImage = train['image']
    trainImage = trainImage.to_numpy()
    # Conversion
    trainImage = np.array([val for val in trainImage])
    trainLabel = label.fit_transform(train['label'])
    trainLabel = to_categorical(trainLabel)
    labels = label.classes_

    valText = bert_encode(val['text'], tokenizer, max_len=max_len)
    valImage = val['image']
    valImage = valImage.to_numpy()
    valImage = np.array([val for val in valImage])
    valLabel = label.fit_transform(val['label'])
    valLabel = to_categorical(valLabel)
    valLabel = valLabel

    print(fold_var)
    file_path = '/model_'+str(fold_var)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=8)
    callbacks_list = [checkpoint,earlystopping]
    history = model.fit(x=[trainText,trainImage],y=trainLabel,epochs=30,batch_size=70,validation_data=([valText, valImage], valLabel), callbacks=callbacks_list, verbose=1)
    model.load_weights(file_path)
    score = model.evaluate([valText, valImage],valLabel, verbose=0)
    scores_loss.append(score[0])
    scores_acc.append(score[1])
    scores_pre.append(score[2])
    scores_rap.append(score[3])
    #result.append(model.predict([testText,testImage]))
    tf.keras.backend.clear_session()
    fold_var += 1
value_min = min(scores_loss)
value_index = scores_loss.index(value_min)
model.load_weights('/model_'+str(value_index))
best_model = model
best_models.append(best_model)
best_model.save_weights('model_weights.h5')
best_model.save('model_keras.h5')
############# FIN ENTRAINEMENT DU MODEL ##########
testText = bert_encode(dataImageTextTest['text'], tokenizer, max_len=max_len)
testImage = dataImageTextTest['image']
testImage = testImage.to_numpy()
testImage = np.array([val for val in testImage])
testLabel = label.fit_transform(dataImageTextTest['label'])
testLabel = to_categorical(testLabel)
testLabel = testLabel

############ DEBUT GRAPHES ET COURBES #############
plot_learning_curve(best_models)
y_pred = predictModel([testText,testImage])
confusionMatrix(testLabel,y_pred)
plotCurve(testLabel,y_pred)
############ FIN DEBUT GRAPES ET COURBES #############
