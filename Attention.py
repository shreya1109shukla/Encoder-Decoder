import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split


class Encoder(tf.keras.Model):

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):

        super().__init__()
        self.lstm_size = lstm_size
        self.embedding = Embedding(input_dim=inp_vocab_size, output_dim=embedding_size, input_length=input_length,
                           mask_zero=True, name="embedding_layer_encoder")
        self.lstm = LSTM(lstm_size, return_state=True, return_sequences=True, name="Encoder_LSTM")

    def call(self,input_sequence,states):
     
      input_embedd                           = self.embedding(input_sequence)
      self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd, initial_state = states)
      return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    
    def initialize_states(self,batch_size):
      
      self.h_0 = tf.zeros((batch_size,self.lstm_size))
      self.c_0 = tf.zeros((batch_size,self.lstm_size))
      return self.h_0,self.c_0


class Attention(tf.keras.layers.Layer):
 
  def __init__(self,scoring_function, att_units):
    super(Attention, self).__init__()
    self.scoring_function = scoring_function
    
      
    if scoring_function == 'general':
      self.W = tf.keras.layers.Dense(att_units)
      
    elif scoring_function == 'concat':
      self.W1 = tf.keras.layers.Dense(att_units)
      self.W2 = tf.keras.layers.Dense(att_units)
      self.V = tf.keras.layers.Dense(1)
      
  
  
  def call(self,decoder_hidden_state,encoder_output):
    
    if self.scoring_function == 'dot':
        decoder_with_time_axis = tf.expand_dims(decoder_hidden_state, 2)
        score = tf.matmul(encoder_output, decoder_with_time_axis)
        
    elif self.scoring_function == 'general':
        decoder_with_time_axis = tf.expand_dims(decoder_hidden_state, 2)
        score = tf.matmul(self.W(encoder_output), decoder_with_time_axis)
        
    elif self.scoring_function == 'concat':
        decoder_with_time_axis = tf.expand_dims(decoder_hidden_state, 1)
        score = self.V(tf.nn.tanh(self.W1(decoder_with_time_axis) + self.W2(encoder_output)))
    

    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * encoder_output
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights


class OneStepDecoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):

      
      super(OneStepDecoder, self).__init__()
      self.embeddings = tf.keras.layers.Embedding(tar_vocab_size,embedding_dim,input_length=input_length)

      self.lstm = tf.keras.layers.LSTM(dec_units,return_sequences=True,return_state=True,)
      self.attention = Attention(score_fun, att_units)
      self.dense = tf.keras.layers.Dense(tar_vocab_size, activation='softmax')

  def call(self,input_to_decoder, encoder_output, state_h,state_c):
    
    context_vector, attention_weights = self.attention(state_h, encoder_output)
   
    x = self.embeddings(input_to_decoder)
    x = tf.concat([x, tf.expand_dims(context_vector, 1)], axis=-1)
    output, decoder_state_h, decoder_state_c = self.lstm(x,initial_state=[state_h, state_c])
    output = tf.reshape(output, (-1, output.shape[2]))
    output = self.dense(output)
    
    return output, decoder_state_h, decoder_state_c, attention_weights,context_vector


class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, output_length, dec_units ,score_fun ,att_units):
      
      super(Decoder, self).__init__()
      self.one_step_decoder = OneStepDecoder(out_vocab_size, embedding_dim, output_length, dec_units, score_fun, att_units)
      self.out_vocab_size = out_vocab_size
      self.output_length = output_length
        
    def call(self, input_to_decoder,encoder_output,decoder_hidden_states,decoder_cell_states):

        
        result = []
        for i in range(len(input_to_decoder)):
          sentence = input_to_decoder[i]
          state_h = decoder_hidden_states[i]
          state_c = decoder_cell_states[i]
          values = encoder_output[i]

          word_outputs = []

          for word in sentence:
            d = tf.expand_dims(word, 0)
            d = tf.expand_dims(d, 0)
            e = tf.expand_dims(values, 0)
            h = tf.expand_dims(state_h, 0)
            c = tf.expand_dims(state_c, 0)
            output, _, _, _, _ = self.one_step_decoder(d, e, h, c)


            for j in output:
              word_outputs.append(j)
        
          result.append(word_outputs)
        
        return tf.convert_to_tensor(result)
        
    
class encoder_decoder(tf.keras.Model):
  def __init__(self,in_vocab_size, embedding_size,enc_units, input_length,out_vocab_size, dec_units,score_fun, att_units, batch_size):
   
    super(encoder_decoder, self).__init__()
    self.batch_size = batch_size
    self.encoder = Encoder(in_vocab_size, embedding_size, enc_units,input_length)
    self.decoder = Decoder(out_vocab_size, embedding_size, input_length, dec_units, score_fun, att_units)
  
  def call(self,data):
    
    ita, eng = data[0], data[1]
    initial_state = self.encoder.initialize_states(self.batch_size)
    encoder_output, state_h, state_c = self.encoder(ita, initial_state)
    output = self.decoder(eng, encoder_output, state_h, state_c)

    return output

def custom_lossfunction(targets,logits):

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)


class Dataset:
    def __init__(self, data, tknizer_ita, tknizer_eng, max_len):
        self.encoder_inps = data['italian'].values
        self.decoder_inps = data['english_inp'].values
        self.decoder_outs = data['english_out'].values
        self.tknizer_eng = tknizer_eng
        self.tknizer_ita = tknizer_ita
        self.max_len = max_len
 
    def __getitem__(self, i):
        self.encoder_seq = self.tknizer_ita.texts_to_sequences([self.encoder_inps[i]])\
        self.decoder_inp_seq = self.tknizer_eng.texts_to_sequences([self.decoder_inps[i]])
        self.decoder_out_seq = self.tknizer_eng.texts_to_sequences([self.decoder_outs[i]])
 
        self.encoder_seq = pad_sequences(self.encoder_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, maxlen=self.max_len, dtype='int32', padding='post')
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq
 
    def __len__(self): 
        return len(self.encoder_inps)
 
    
class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))
 
 
    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
 
        batch = [np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)]
        return tuple([[batch[0],batch[1]],batch[2]])
 
    def __len__(self):  
        return len(self.indexes) // self.batch_size
 
    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)


train_dataset = Dataset(train, tknizer_ita, tknizer_eng, 20)
test_dataset  = Dataset(validation, tknizer_ita, tknizer_eng, 20)

x = np.squeeze(train_dataset, axis=2)
y = np.squeeze(test_dataset , axis = 2)

ita_train,eng_in_train,eng_out_train = x[:,0,:] , x[:,1,:], x[:,2,:]
ita_test,eng_in_test , eng_out_test = y[:,0,:] , y[:,1,:], y[:,2,:]

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True, reduction='none')

encoder_decoder = encoder_decoder(vocab_size_ita,216,1,(None, 20),vocab_size_eng,1,'dot',1,1024)