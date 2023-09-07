import spacy
import pysbd
from tensorflow.keras.preprocessing.text import Tokenizer
from spacy.lang.en import English
import spacy
import wget
#nltk.download('omw-1.4')
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('PorterStemmer')
import contractions
from bs4 import BeautifulSoup
#import numpy as np
#import re
import tqdm
import unicodedata
from numpy import asarray
from numpy import zeros
import inflect
import numpy as np
import pandas as pd
import re
import io
import statistics
import unicodedata
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem import PorterStemmer
#import tensorflow_decision_forests as tfdf
#from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.keras.layers import TextVectorization,BatchNormalization
#from keras.layers.embeddings import Embedding
from tensorflow.keras import layers
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional,LSTM, Add,GRU,MaxPooling1D, GlobalMaxPool1D, GlobalMaxPooling1D, Dropout,Conv1D,Embedding,Flatten, Input, Layer,GlobalAveragePooling1D,Activation,Lambda,LayerNormalization, ConvLSTM1D, Concatenate, Average,AlphaDropout,Reshape, multiply,CuDNNLSTM,SpatialDropout1D
from tensorflow import keras
#import tensorflow_decision_forests as tfdf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf 
import tensorflow_hub as hub
#import chardet
from sklearn import preprocessing
from sklearn import metrics
import nltk
import keras.backend as K
nltk.download('punkt')

def preprocessing(data):
  ligne =[]
  p = inflect.engine()
  seg = pysbd.Segmenter(language="en", clean=False)
  nlp = English()
  tokenizer = nlp.tokenizer
  RE = "(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)|(RT)"
  ps = PorterStemmer()
  i=0
  for sentences in data : 
    #mitext = ''
    mitext = []
    for sentence in seg.segment(sentences):
      filtered_sentence = []
      #re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)|(RT)","",sentence)
      for word in [token.text for token in tokenizer(sentence)] :
        #if word.isdigit():
            #word = p.number_to_words(word)
        match = re.search(RE, word)
        capital = word.title()
        if (match == None) or (word==capital):
          filtered_sentence.append(word)
      tokens_tag = pos_tag(filtered_sentence)
      sentenceTag = []
      for word in tokens_tag :
          #sentenceTag.append(word[0])
          if (word[1] in ["NNP","JJ","VB"]) and (len(word[0])>2) :
            sentenceTag.append(word[0])
      filtered_sentenceOtre = [word for word in sentenceTag if word.lower() not in stopwords.words('english')]
      stems = []
      for word in filtered_sentenceOtre:
          stem = ps.stem(word)
          stems.append(stem)
      text = ' '.join([x for x in stems])
      #mitext = mitext+text+' '
      mitext.append(text)
    i=i+1
    #print(i)
    texts =' '.join([x for x in mitext])
    ligne.append(texts)
    #print (mitext)
  return ligne



def strip_html_tags(text):
  soup = BeautifulSoup(text, "html.parser")
  [s.extract() for s in soup(['iframe', 'script'])]
  stripped_text = soup.get_text()
  stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
  return stripped_text

def remove_accented_chars(text):
  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return text

def preProcessCorpus(docs):
  norm_docs = []
  stop_words = set(stopwords.words('english'))
  for doc in tqdm.tqdm(docs):
    #doc = strip_html_tags(doc)
    doc = doc.lower()
    doc = doc.translate(doc.maketrans("\n\t", "  "))
    #doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = re.sub(r'[0-9]+', '', doc)
    word_tokens = word_tokenize(doc)
    doc = [w for w in word_tokens if not w.lower() in stop_words]
    doc = ' '.join([x for x in doc])
    doc = doc.strip()  
    norm_docs.append(doc)
  return norm_docs
def preprocessing2(norm_data):
  t = Tokenizer(oov_token='<UNK>')
  # fit the tokenizer on the documents
  t.fit_on_texts(norm_data)
  t.word_index['<PAD>'] = 0
  data_sequences = t.texts_to_sequences(norm_data)
  MAX_SEQUENCE_LENGTH = 300
  X_train = tf.keras.preprocessing.sequence.pad_sequences(data_sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
  #VOCAB_SIZE = len(t.word_index)
  return X_train
 
def preprocessingFit(data):
      ligne =[]
      p = inflect.engine()
      #seg = pysbd.Segmenter(language="en", clean=False)
      nlp = English()
      tokenizer = nlp.tokenizer
      RE = "(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)|(RT)"
      ps = PorterStemmer()
      i=0
      stop_words = set(stopwords.words('english'))
      for sentences in data : 
        #sentences = re.sub(r"\b\d+\b", "", sentences)
        #sentences = re.sub(r'[^\w\s]','',sentences)
        #punctuations="?:!.,;"
        #myStopWord=[a,about,an,are,as,at,be,by,for,from,how,in,is,of,on,or,that,the,these,this,too,was,what,when,where,who,will]
        sentence_words = nltk.word_tokenize(sentences)
        
        tokens_without_sw = [word for word in sentence_words if not word.lower() in stop_words]
        '''tokens_tag = pos_tag(tokens_without_sw)
        sentenceTag = []
        for word in tokens_tag :
            if word[1] in ["NNP","JJ","VB"]:
                sentenceTag.append(word[0])'''
        filtered_sentence = []
        for word in tokens_without_sw:
          match = re.search(RE, word)
          capital = word.title()
          if (match == None):
              if(len(word)>2):
                  word = word.lower()
                  filtered_sentence.append(word)
          else:
              if(word==capital):
                word = word.lower()
                filtered_sentence.append(word)
        sentence = ' '.join([x for x in filtered_sentence])
        ligne.append(sentence)
      return ligne
def preprocessingTest(data):
      ligne =[]
      p = inflect.engine()
      #seg = pysbd.Segmenter(language="en", clean=False)
      nlp = English()
      tokenizer = nlp.tokenizer
      RE = "(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)|(RT)"
      ps = PorterStemmer()
      stop_words = set(stopwords.words('english'))
      i=0
      for sentences in data :
        #sentences = re.sub(r"\b\d+\b", "", sentences)
        #sentences = re.sub(r'[^\w\s]','',sentences)
        punctuations="?:!.,;"
        #myStopWord=[a,about,an,are,as,at,be,by,for,from,how,in,is,of,on,or,that,the,these,this,too,was,what,when,where,who,will]
        sentence_words = nltk.word_tokenize(sentences)
        tokens_without_sw = [word for word in sentence_words if not word.lower() in stop_words]
        '''tokens_tag = pos_tag(tokens_without_sw)
        sentenceTag = []
        for word in tokens_tag :
            if word[1] in ["NNP","JJ","VB"]:
                sentenceTag.append(word[0])'''
        filtered_sentence = []
        for word in tokens_without_sw:
          match = re.search(RE, word)
          capital = word.title()
          if (match == None):
              if(len(word)>2):
                  word = word.lower() 
                  filtered_sentence.append(ps.stem(word))
          else:
              if(word==capital):
                  word = word.lower()
                  filtered_sentence.append(ps.stem(word))
        sentence = ' '.join([x for x in filtered_sentence])
        ligne.append(sentence)
      return ligne
#def splitData(data):      

def cnn_bilstm(word_index, embedding_matrix,EMBEDDING_DIM=100,filters=32,kernel_size=2):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)(input)
  model = Conv1D(filters=128,kernel_size=5, activation='relu')(embedding_layer)
  model = MaxPooling1D(2)(model)
  model = Bidirectional(LSTM(32))(model)
  out = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=out)
  #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
  return model
def SqueezeExcite(_input, r=4):  # r == "reduction factor"; see paper
    filters = K.int_shape(_input)[-1]

    se = GlobalAveragePooling1D()(_input)
    se = Reshape((1, filters))(se)
    se = Dense(filters//r, activation='relu',use_bias=False)(se)
    se = Dense(filters,activation='sigmoid', use_bias=False,kernel_initializer='he_normal')(se)
    return multiply([_input, se])
    
def deep_cnn_bilstm(word_index, embedding_matrix,EMBEDDING_DIM=100):
  #input = Input(shape=(429,),dtype='int32')
  input = Input(shape=(429,),dtype='int32')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)(input)
  model0 = Dropout(0.7)(embedding_layer)
  model1 = Conv1D(filters=128,kernel_size=5,activation='elu')(model0)
  model1 = BatchNormalization()(model1)
  model1 = Dropout(0.5)(model1)
  model1 = MaxPooling1D(2)(model1)
  #model1 = LSTM(32)(model1)
  model1 = Bidirectional(LSTM(32))(model1)
  #model1 = Dropout(0.5)(model1)
  
 
  model2 = Conv1D(filters=128,kernel_size=3,activation='elu')(model0)
  model2 = BatchNormalization()(model2)
  model2 = Dropout(0.5)(model2)
  model2 = MaxPooling1D(2)(model2)
  #model2 = LSTM(32)(model2)
  model2 = Bidirectional(LSTM(32))(model2)
  #model2 = Dropout(0.5)(model2)
  
  out = Concatenate()([model1,model2])
  out = Dense(32,activation='relu')(out)
  #out = Dropout(0.7)(out)
  out = Dense(1,activation='sigmoid')(out)
  model = keras.Model(inputs=input,outputs=out)
  return model  

def deep_cnn_lstm(word_index, embedding_matrix,EMBEDDING_DIM=100,filters=32,kernel_size=2):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(429,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)(input)
  model0 = Dropout(0.7)(embedding_layer)
  model1 = Conv1D(filters=128,kernel_size=5,activation='elu')(model0)
  model1 = BatchNormalization()(model1)
  model1 = Dropout(0.5)(model1)
  model1 = MaxPooling1D(2)(model1)
  model1 = LSTM(32)(model1)
  
  model2 = Conv1D(filters=128,kernel_size=3,activation='elu')(model0)
  model2 = BatchNormalization()(model2)
  model2 = Dropout(0.5)(model2)
  model2 = MaxPooling1D(2)(model2)
  model2 = LSTM(32)(model2)
  
  out = Concatenate()([model1,model2])
  out = Dense(32,activation='relu')(out)
  out = Dense(1,activation='sigmoid')(out)
  model = keras.Model(inputs=input,outputs=out)
  return model  

def deep_cnn_gru(word_index, embedding_matrix,EMBEDDING_DIM=100,filters=32,kernel_size=2):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)(input)
  model = Conv1D(filters=32,kernel_size=3, activation='relu')(embedding_layer)
  model = MaxPooling1D(2)(model)
  model = Conv1D(filters=64,kernel_size=5, activation='relu')(model)
  model = MaxPooling1D(2)(model)
  model = GRU(32,return_sequences=True)(model)
  model = GlobalMaxPooling1D()(model)
  out = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=out)
  return model  
  
def cnn_gru(word_index, embedding_matrix,EMBEDDING_DIM=100):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  model = Conv1D(128, 5,activation='relu')(embedding_layer)
  model = MaxPooling1D(2)(model)
  model = GRU(32)(model)
  model = Flatten()(model)
  model = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=model)
  return model

def gru(word_index, embedding_matrix,EMBEDDING_DIM=100) :
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  model = GRU(32)(embedding_layer)
  model = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=model)
  return model
  
def lstm(word_index, embedding_matrix,EMBEDDING_DIM=100) :
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  model = LSTM(32)(embedding_layer)
  model = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=model)
  return model

def bilstm(word_index, embedding_matrix,EMBEDDING_DIM=100) :
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  model = Bidirectional(LSTM(32))(embedding_layer)
  model = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=model)
  return (model)
  
def cnn(word_index, embedding_matrix,EMBEDDING_DIM=100) :
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  model = Conv1D(128, 5,activation='relu')(embedding_layer)
  model = MaxPooling1D(2)(model)
  model = Flatten()(model)
  model = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=model)
  return model
 
def cnn_lstm(word_index, embedding_matrix,EMBEDDING_DIM=100):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  model = Conv1D(128, 5,activation='relu')(embedding_layer)
  model = MaxPooling1D(2)(model)
  model = LSTM(32)(model)
  model = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=model)
  #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
  return model

def cnn_rdf(word_index, embedding_matrix,EMBEDDING_DIM=100):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  model = Conv1D(128, 5,activation='relu')(model)
  model = MaxPooling1D(2)(model)
  #model = Dropout(0.1)(model)
  model = Dropout(0.1)(model)
  model = BatchNormalization()(model)
  model = Conv1D(128, 5,activation='relu')(model)
  model = MaxPooling1D(2)(model)
  #model = Dropout(0.1)(model)
  model = Dropout(0.1)(model)
  model = BatchNormalization()(model)
  model = Conv1D(256, 5,activation='relu')(model)
  model = MaxPooling1D(2)(model)
  model = Dropout(0.1)(model)
  model = BatchNormalization()(model)
  model = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=model)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
  return model 

def cnn_bilstm_Adabbost(word_index, embedding_matrix,EMBEDDING_DIM=100):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
  
  model = Sequential()
  model.add(Embedding(len(word_index)+1, EMBEDDING_DIM, input_length=300))
  model.add(Conv1D(32, 5,activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(Conv1D(64, 3,activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(Conv1D(128, 5,activation='relu'))
  model.add(MaxPooling1D(2))
  model.add(LSTM(64,return_sequences=True))
  model.add(Bidirectional(LSTM(64)))
  model.add(Dense(32,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
  return model 
def cnn_lstm1(word_index, embedding_matrix,EMBEDDING_DIM=100):
    
    optimizer = tf.keras.optimizers.Adam()
    input = Input(shape=(300,), dtype='int64')
    embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False,)(input)
    model = Conv1D(128, 5,activation='relu')(embedding_layer)
    model = MaxPooling1D(2)(model)
    model = LSTM(32)(model)
    lastLayer = Dense(1,activation='sigmoid')(model)
    model = keras.Model(inputs=input,outputs=lastLayer)
    nn_without_head = tf.keras.models.Model(inputs=model.inputs, outputs=lastLayer)
    df_and_nn_model = tfdf.keras.RandomForestModel(preprocessing=nn_without_head,num_trees=300)
    
    return (model)

def cnn_cnn(word_index, embedding_matrix,EMBEDDING_DIM=100,filters=32,kernel_size=2):
  optimizer = tf.keras.optimizers.Adam()
  input = Input(shape=(300,), dtype='int64')
  embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)(input)
  model = Conv1D(filters=32,kernel_size=3, activation='elu')(embedding_layer)
  model = MaxPooling1D(2)(model)
  model = Conv1D(filters=64,kernel_size=5, activation='elu')(model)
  model = GlobalMaxPooling1D()(model)
  model = Flatten()(model)
  model = Dense(units=2,activation='relu')(model)
  out = Dense(1,activation='sigmoid')(model)
  model = keras.Model(inputs=input,outputs=out)
  #nn_without_head = tf.keras.models.Model(inputs=input,outputs=model)
  #df_and_nn_model = tfdf.keras.RandomForestModel(preprocessing=nn_without_head,num_trees=300)
  #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
  return model
  
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights
class CustomConstraint(Constraint):
    n=5
    
    def __init__(self,k,s):
        self.k = k
        self.s = s
        np.random.rand2 = lambda *args, dtype=np.float32: np.random.rand(*args).astype(dtype)
        f1 = np.random.rand2(s,s)
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
def fake_virtual(k=3,s=5):
    input1 = Input(shape=(50,))
    embedding_layer = Embedding(len(word_index)+1,100,embeddings_initializer=keras.initializers.Constant(embedding_matrix),trainable=False)(input1)
    model1 = Bidirectional(LSTM(32))(embedding_layer)
    model1 = Dense(64, activation='relu')(model1)
    input2 = Input(shape=(224,224,3))
    conv_layer = Conv2D(filters=k, kernel_size=s,kernel_constraint=CustomConstraint(k,s), padding='same', use_bias=False)(input2)
    model = Conv2D(filters=16,kernel_size=3, padding='same',use_bias=False)(conv_layer)
    model = BatchNormalization(axis=3, scale=False)(model)
    model = Activation('relu')(model)
    model = GlobalAveragePooling2D()(model)
    # Fully connected layers
    model = Dense(64, activation='relu')(model)
    concat = layers.Concatenate()([model1,model])
    
    final_model_output = Dense(2, activation='softmax')(concat)
    final_model = Model(inputs=[input1,input2], outputs=final_model_output)
    final_model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", f1_m])
    return final_model  
