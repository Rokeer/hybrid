# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   5/22/20 4:19 PM

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding, Dropout, Reshape, TimeDistributed, Conv1D, LSTM, Dense, Concatenate, AdditiveAttention, Attention
from tensorflow.keras.initializers import Constant
from networks.zeromasking import ZeroMaskedEntries
from networks.softattention import SelfAttention


class WordWord(layers.Layer):
    def __init__(self, N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters, **kwargs):
        super(WordWord, self).__init__(**kwargs)
        self.feat_embedding = feat_embedding
        self.model_parameters = model_parameters
        self.N = N
        self.L = L
        self.dropout = dropout
        self.nbfilters = nbfilters
        self.filter1_len = filter1_len

        if self.feat_embedding > 0:
            self.embedding = Embedding(output_dim=self.feat_embedding, input_dim=self.model_parameters['vocab_size'],
                                       input_length=self.N * self.L, mask_zero=True, name='fx')
            self.zero_masked_entries = ZeroMaskedEntries(name='zx_maskedout')
            self.drop = Dropout(self.dropout, name='drop_zx')
            self.reshape = Reshape((self.N, self.L, self.feat_embedding), name='resh_fW')
        elif self.feat_embedding < 0:
            if 'emb_table' in self.model_parameters:
                self.embedding = Embedding(output_dim=self.model_parameters['output_dim'],
                                           input_dim=self.model_parameters['vocab_size'],
                                           input_length=self.model_parameters['f_max_sentnum'] * self.model_parameters['f_max_sentlen'],
                                           embeddings_initializer=Constant(self.model_parameters['emb_table']),
                                           trainable=False,
                                           mask_zero=True,
                                           name='fx')
                self.zero_masked_entries = ZeroMaskedEntries(name='zx_maskedout')
                self.drop = Dropout(self.dropout, name='drop_zx')
                self.reshape = Reshape((self.model_parameters['f_max_sentnum'], self.model_parameters['f_max_sentlen'],
                                        self.model_parameters['output_dim']), name='resh_fW')
            elif 'use_same_emb' in self.model_parameters:
                self.embedding = self.model_parameters['use_same_emb']
                self.zero_masked_entries = ZeroMaskedEntries(name='zx_maskedout')
                self.drop = Dropout(self.dropout, name='drop_zx')
                self.reshape = Reshape((self.model_parameters['f_max_sentnum'], self.model_parameters['f_max_sentlen'],
                                        self.model_parameters['output_dim']), name='resh_fW')

        self.time_distributed = TimeDistributed(Conv1D(self.nbfilters, self.filter1_len, padding='valid'), name='ext_zcnn')

    def call(self, inputs):
        if self.feat_embedding > 0:
            fx = self.embedding(inputs)
            fx_maskedout = self.zero_masked_entries(fx)
            drop_fx = self.drop(fx_maskedout)
            resh_fW = self.reshape(drop_fx)
        elif self.feat_embedding == 0:
            resh_fW = inputs
        else:
            if 'emb_table' in self.model_parameters:
                fx = self.embedding(inputs)
                fx_maskedout = self.zero_masked_entries(fx)
                drop_fx = self.drop(fx_maskedout)
                resh_fW = self.reshape(drop_fx)
            elif 'use_same_emb' in self.model_parameters:
                fx = self.embedding(inputs)
                fx_maskedout = self.zero_masked_entries(fx)
                drop_fx = self.drop(fx_maskedout)
                resh_fW = self.reshape(drop_fx)
            else:
                resh_fW = inputs

        ext_zcnn = self.time_distributed(resh_fW)

        return ext_zcnn


class SentWord(layers.Layer):
    def __init__(self, N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters, lstm_units, **kwargs):
        super(SentWord, self).__init__(**kwargs)
        self.word_word = WordWord(N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters)
        self.lstm_units = lstm_units
        self.time_distributed = TimeDistributed(SelfAttention(), name='ext_avg_zcnn')
        self.lstm = LSTM(self.lstm_units, return_sequences=True, name='ext_hz_lstm')

    def call(self, inputs):
        ext_zcnn = self.word_word(inputs)
        ext_avg_zcnn = self.time_distributed(ext_zcnn)
        ext_hz_lstm = self.lstm(ext_avg_zcnn)
        return ext_hz_lstm


class DocWord(layers.Layer):
    def __init__(self, N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters, lstm_units, **kwargs):
        super(DocWord, self).__init__(**kwargs)
        self.sent_word = SentWord(N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters, lstm_units)
        self.self_attention = SelfAttention(name='ext_avg_hz_lstm')

    def call(self, inputs):
        ext_hz_lstm = self.sent_word(inputs)
        ext_avg_hz_lstm = self.self_attention(ext_hz_lstm)
        return ext_avg_hz_lstm


class SentSent(layers.Layer):
    def __init__(self, N, feat_embedding, dropout, model_parameters, lstm_units, **kwargs):
        super(SentSent, self).__init__(**kwargs)
        self.N = N
        self.feat_embedding = feat_embedding
        self.model_parameters = model_parameters
        self.dropout = dropout
        self.lstm_units = lstm_units

        if self.feat_embedding > 0:
            self.embedding = Embedding(output_dim=self.feat_embedding, input_dim=self.model_parameters['vocab_size'], input_length=self.N, mask_zero=True, name='fx')
            self.zero_masked_entries = ZeroMaskedEntries(name='zx_maskedout')
            self.drop = Dropout(self.dropout, name='drop_zx')
        self.lstm = LSTM(self.lstm_units, return_sequences=True, name='ext_hz_lstm')

    def call(self, inputs):
        if self.feat_embedding > 0:
            fx = self.embedding(inputs)
            fx_maskedout = self.zero_masked_entries(fx)
            drop_fx = self.drop(fx_maskedout)
            tmp_feat_input = drop_fx
        elif self.feat_embedding == 0:
            tmp_feat_input = inputs
        else:
            if 'emb_table' in self.model_parameters:
                raise NotImplementedError
            else:
                tmp_feat_input = inputs

        ext_hz_lstm = self.lstm(tmp_feat_input)
        return ext_hz_lstm


class DocSent(layers.Layer):
    def __init__(self, N, feat_embedding, dropout, model_parameters, lstm_units, **kwargs):
        super(DocSent, self).__init__(**kwargs)
        self.sent_sent = SentSent(N, feat_embedding, dropout, model_parameters, lstm_units)
        self.self_attention = SelfAttention(name='ext_avg_hz_lstm')

    def call(self, inputs):
        ext_hz_lstm = self.sent_sent(inputs)
        ext_avg_hz_lstm = self.self_attention(ext_hz_lstm)
        return ext_avg_hz_lstm


class DocDoc(layers.Layer):
    def __init__(self, **kwargs):
        super(DocDoc, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

