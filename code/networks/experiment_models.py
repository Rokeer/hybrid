# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/17/19 9:53 AM

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Reshape, TimeDistributed, Conv1D, LSTM, Dense, Concatenate, AdditiveAttention, Attention
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adagrad, RMSprop

from networks.zeromasking import ZeroMaskedEntries
from networks.softattention import SelfAttention
from networks.matrix_attention import MatrixAttention
from networks.masked_softmax import MaskedSoftmax
from networks.weighted_sum import WeightedSum
from networks.max import Max
from networks.repeat_like import RepeatLike
from networks.complex_concat import ComplexConcat
# from networks.transformers_layer import AlbertLayer
from networks.integration_layers import WordWord, SentWord, DocWord, SentSent, DocSent, DocDoc


from utils import get_logger
import tensorflow.keras.backend as K
import numpy as np
import time

logger = get_logger("Build experimental model")

all_modes = [
    'att_doc_doc',
    'att_doc_sent',
    'att_doc_word',
    'att_sent_sent',
    'att_sent_word',
    'att_word_word'
]

def get_optimizer(name, lr):
    if name == 'sgd':
        return SGD(lr=lr)
    elif name == 'adagrad':
        return Adagrad(lr=lr)
    elif name == 'rmsprop':
        return RMSprop(lr=lr)
    else:
        raise NotImplementedError


def compile_model(model, opts):
    optimizer = get_optimizer(opts.optimizer, opts.learning_rate)
    start_time = time.time()
    model.compile(loss='mse', optimizer=optimizer)
    total_time = time.time() - start_time
    logger.info("Model compiled in %.4f s" % total_time)


def model_layer(mode, name, N, L, feat_embedding, dropout, nbfilters, filter1_len, lstm_units, model_parameters):
    mode_name = mode[1] + mode[2]
    if mode_name == 'wordword':
        return WordWord(N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters, name=name)
    elif mode_name == 'sentword':
        return SentWord(N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters, lstm_units, name=name)
    elif mode_name == 'docword':
        return DocWord(N, L, feat_embedding, dropout, nbfilters, filter1_len, model_parameters, lstm_units, name=name)
    elif mode_name == 'sentsent':
        return SentSent(N, feat_embedding, dropout, model_parameters, lstm_units, name=name)
    elif mode_name == 'docsent':
        return DocSent(N, feat_embedding, dropout, model_parameters, lstm_units, name=name)
    elif mode_name == 'docdoc':
        return DocDoc(name=name)

def build_hrcnn_combined_model(
        opts,
        model_configs,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Word level Integration, Word level Features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s, concatenation_mode = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout, opts.concatenation_mode))


    emb = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')


    feature_inputs = []
    feature_outputs = {
        'word': [],
        'word_name': [],
        'word_concat_mode': [],
        'sent': [],
        'sent_name': [],
        'sent_concat_mode': [],
        'doc': [],
        'doc_name': [],
        'doc_concat_mode': [],
    }
    for config in model_configs:
        name = config['feature']
        mode = config['mode'].split('_')
        model_parameters = config['model_parameters']
        if mode[2] == 'word':
            if config['embedding_size'] > 0:
                feature_input = Input(shape=(N * L,), dtype='int32', name=name+'feature_input')
            elif config['embedding_size'] == 0:
                feature_input = Input(shape=(N, L, model_parameters['vocab_size'],), dtype='float32', name=name+'feature_input')
            else:
                if 'emb_table' in model_parameters:
                    feature_input = Input(shape=(model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],), dtype='int32', name=name+'feature_input')
                elif 'use_same_emb' in model_parameters:
                    model_parameters['use_same_emb'] = emb
                    feature_input = Input(shape=(model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],), dtype='int32', name=name+'feature_input')
                else:
                    feature_input = Input(shape=(N, L, model_parameters['feature_len'],), dtype='float32', name=name+'feature_input')
        if mode[2] == 'sent':
            if config['embedding_size'] > 0:
                feature_input = Input(shape=(N,), dtype='int32', name=name+'feature_input')
            elif config['embedding_size'] == 0:
                feature_input = Input(shape=(N, model_parameters['vocab_size'],), dtype='float32', name=name+'feature_input')
            else:
                if 'emb_table' in model_parameters:
                    raise NotImplementedError
                elif 'use_same_emb' in model_parameters:
                    raise NotImplementedError
                else:
                    feature_input = Input(shape=(N, model_parameters['feature_len'],), dtype='float32',
                                          name=name+'feature_input')
        if mode[2] == 'doc':
            feature_input = Input(shape=(model_parameters['feature_len'],), dtype='float32', name=name+'feature_input')

        feature_inputs.append(feature_input)
        feature_output = model_layer(mode, name, N, L, config['embedding_size'], opts.dropout, opts.nbfilters, opts.filter1_len, opts.lstm_units, model_parameters)(feature_input)
        feature_outputs[mode[1]].append(feature_output)
        feature_outputs[mode[1] + '_name'].append(name)
        feature_outputs[mode[1] + '_concat_mode'].append(mode[0])




    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = emb(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)


    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    if opts.concatenation_mode == 'config':
        concat_layers = [zcnn]
        for word_output, word_name, word_concat_mode in zip(feature_outputs['word'], feature_outputs['word_name'], feature_outputs['word_concat_mode']):
            if word_concat_mode == 'concat':
                logger.info(f"Using Simple Concatenation")
                concat_layers.append(word_output)
            elif word_concat_mode == 'att':
                logger.info("Using Attention")
                concat_layers.append(Attention(name=word_name + '_attn_concat')([zcnn, word_output]))
            elif word_concat_mode == 'co':
                logger.info("Using Co-Attention")
                logger.warn("Please remind that the co-attention concat only supports sentence level integration, so word level concat is passed")
            else:
                raise NotImplementedError
        concat_zcnn = K.concatenate(concat_layers, axis=-1)
    else:
        if opts.concatenation_mode == 'concat':
            logger.info(f"Using Simple Concatenation")
            concat_zcnn = K.concatenate([zcnn] + feature_outputs['word'], axis=-1)
        elif opts.concatenation_mode == 'attn':
            logger.info("Using Attention")
            concat_layers = [zcnn]
            for word_output, word_name in zip(feature_outputs['word'], feature_outputs['word_name']):
                concat_layers.append(Attention(name=word_name+'_attn_concat')([zcnn, word_output]))
            concat_zcnn = K.concatenate(concat_layers, axis=-1)
        elif opts.concatenation_mode == 'co':
            logger.info("Using Co-Attention")
            logger.warn("Please remind that the co-attention concat only supports sentence level integration, so word level concat is passed")
            concat_layers = [zcnn]
            concat_zcnn = K.concatenate(concat_layers, axis=-1)
        else:
            raise NotImplementedError

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(concat_zcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)


    if opts.concatenation_mode == 'config':
        concat_layers = [hz_lstm]
        for sent_output, sent_name, sent_concat_mode in zip(feature_outputs['sent'], feature_outputs['sent_name'], feature_outputs['sent_concat_mode']):
            if sent_concat_mode == 'concat':
                logger.info(f"Using Simple Concatenation")
                concat_layers.append(sent_output)
            elif sent_concat_mode == 'att':
                logger.info("Using Attention")
                concat_layers.append(Attention(name=sent_name + '_attn_concat')([hz_lstm, sent_output]))
            elif sent_concat_mode == 'co':
                logger.info("Using Co-Attention")
                logger.warn("Please remind that the co-attention concat only supports sentence level integration")
                # PART 2:
                # Now we compute a similarity between the passage words and the question words, and
                # normalize the matrix in a couple of different ways for input into some more layers.
                matrix_attention_layer = MatrixAttention(name=sent_name + '_essay_context_similarity')
                # Shape: (batch_size, num_passage_words, num_question_words)
                essay_context_similarity = matrix_attention_layer([hz_lstm, sent_output])

                # Shape: (batch_size, num_passage_words, num_question_words), normalized over question
                # words for each passage word.
                essay_context_attention = MaskedSoftmax()(essay_context_similarity)
                weighted_sum_layer = WeightedSum(name=sent_name + "_essay_context_vectors", use_masking=False)
                # Shape: (batch_size, num_passage_words, embedding_dim * 2)
                weighted_hz_lstm = weighted_sum_layer([sent_output, essay_context_attention])

                # Min's paper finds, for each document word, the most similar question word to it, and
                # computes a single attention over the whole document using these max similarities.
                # Shape: (batch_size, num_passage_words)
                context_essay_similarity = Max(axis=-1)(essay_context_similarity)
                # Shape: (batch_size, num_passage_words)
                context_essay_attention = MaskedSoftmax()(context_essay_similarity)
                # Shape: (batch_size, embedding_dim * 2)
                weighted_sum_layer = WeightedSum(name=sent_name + "_context_essay_vector", use_masking=False)
                context_essay_vector = weighted_sum_layer([hz_lstm, context_essay_attention])

                # Then he repeats this question/passage vector for every word in the passage, and uses it
                # as an additional input to the hidden layers above.
                repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
                # Shape: (batch_size, num_passage_words, embedding_dim * 2)
                tiled_context_essay_vector = repeat_layer([context_essay_vector, hz_lstm])

                # Please note that, the combination is different than the original code
                # because we will cocat hz_lstm manually at the end
                complex_concat_layer = ComplexConcat(combination='2,1*2,1*3', name=sent_name + '_final_merged_passage')
                final_merged_passage = complex_concat_layer([hz_lstm,
                                                             weighted_hz_lstm,
                                                             tiled_context_essay_vector])
                concat_layers.append(final_merged_passage)
            else:
                raise NotImplementedError
        concat_hz_lstm = K.concatenate(concat_layers, axis=-1)
    else:
        if opts.concatenation_mode == 'concat':
            logger.info(f"Using Simple Concatenation")
            concat_hz_lstm = K.concatenate([hz_lstm] + feature_outputs['sent'], axis=-1)
        elif opts.concatenation_mode == 'attn':
            logger.info("Using Attention")
            concat_layers = [hz_lstm]
            for sent_output, sent_name in zip(feature_outputs['sent'], feature_outputs['sent_name']):
                concat_layers.append(Attention(name=sent_name+'_attn_concat')([hz_lstm, sent_output]))
            concat_hz_lstm = K.concatenate(concat_layers, axis=-1)
        elif opts.concatenation_mode == 'co':
            logger.info("Using Co-Attention")
            logger.info("Please remind that the co-attention concat only supports sentence level integration")
            concat_layers = [hz_lstm]
            for sent_output, sent_name in zip(feature_outputs['sent'], feature_outputs['sent_name']):
                # PART 2:
                # Now we compute a similarity between the passage words and the question words, and
                # normalize the matrix in a couple of different ways for input into some more layers.
                matrix_attention_layer = MatrixAttention(name=sent_name + '_essay_context_similarity')
                # Shape: (batch_size, num_passage_words, num_question_words)
                essay_context_similarity = matrix_attention_layer([hz_lstm, sent_output])

                # Shape: (batch_size, num_passage_words, num_question_words), normalized over question
                # words for each passage word.
                essay_context_attention = MaskedSoftmax()(essay_context_similarity)
                weighted_sum_layer = WeightedSum(name=sent_name + "_essay_context_vectors", use_masking=False)
                # Shape: (batch_size, num_passage_words, embedding_dim * 2)
                weighted_hz_lstm = weighted_sum_layer([sent_output, essay_context_attention])

                # Min's paper finds, for each document word, the most similar question word to it, and
                # computes a single attention over the whole document using these max similarities.
                # Shape: (batch_size, num_passage_words)
                context_essay_similarity = Max(axis=-1)(essay_context_similarity)
                # Shape: (batch_size, num_passage_words)
                context_essay_attention = MaskedSoftmax()(context_essay_similarity)
                # Shape: (batch_size, embedding_dim * 2)
                weighted_sum_layer = WeightedSum(name=sent_name + "_context_essay_vector", use_masking=False)
                context_essay_vector = weighted_sum_layer([hz_lstm, context_essay_attention])

                # Then he repeats this question/passage vector for every word in the passage, and uses it
                # as an additional input to the hidden layers above.
                repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
                # Shape: (batch_size, num_passage_words, embedding_dim * 2)
                tiled_context_essay_vector = repeat_layer([context_essay_vector, hz_lstm])

                # Please note that, the combination is different than the original code
                # because we will cocat hz_lstm manually at the end
                complex_concat_layer = ComplexConcat(combination='2,1*2,1*3', name=sent_name + '_final_merged_passage')
                final_merged_passage = complex_concat_layer([hz_lstm,
                                                             weighted_hz_lstm,
                                                             tiled_context_essay_vector])
                concat_layers.append(final_merged_passage)
            concat_hz_lstm = K.concatenate(concat_layers, axis=-1)
        else:
            raise NotImplementedError

    if opts.use_co_model:
        avg_hz_lstm = LSTM(opts.lstm_units, return_sequences=False, name='avg_hz_lstm')(concat_hz_lstm)
    else:
        logger.info('Use attention-pooling on text')
        avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(concat_hz_lstm)


    if opts.concatenation_mode == 'config':
        concat_layers = [avg_hz_lstm]
        for doc_output, doc_name, doc_concat_mode in zip(feature_outputs['doc'], feature_outputs['doc_name'], feature_outputs['doc_concat_mode']):
            if doc_concat_mode == 'concat':
                logger.info(f"Using Simple Concatenation")
                concat_layers.append(doc_output)
            elif doc_concat_mode == 'att':
                logger.info("Using Attention")
                concat_layers.append(K.squeeze(Attention(name=doc_name+'_attn_concat')([K.expand_dims(avg_hz_lstm), K.expand_dims(doc_output)]), axis=-1))
            elif doc_concat_mode == 'co':
                logger.info("Using Co-Attention")
                logger.warn("Please remind that the co-attention concat only supports sentence level integration, so word level concat is passed")
            else:
                raise NotImplementedError
        concat_avg_hz_lstm = K.concatenate(concat_layers, axis=-1)
    else:
        if opts.concatenation_mode == 'concat':
            logger.info(f"Using Simple Concatenation")
            concat_avg_hz_lstm = K.concatenate([avg_hz_lstm] + feature_outputs['doc'], axis=-1)
        elif opts.concatenation_mode == 'attn':
            logger.info("Using Attention")
            concat_layers = [avg_hz_lstm]
            for doc_output, doc_name in zip(feature_outputs['doc'], feature_outputs['doc_name']):
                concat_layers.append(K.squeeze(Attention(name=doc_name+'_attn_concat')([K.expand_dims(avg_hz_lstm), K.expand_dims(doc_output)]), axis=-1))
            concat_avg_hz_lstm = K.concatenate(concat_layers, axis=-1)
        elif opts.concatenation_mode == 'co':
            logger.info("Using Co-Attention")
            logger.warn(
                "Please remind that the co-attention concat only supports sentence level integration, so doc level concat is passed")
            concat_layers = [avg_hz_lstm]
            concat_avg_hz_lstm = K.concatenate(concat_layers, axis=-1)
        else:
            raise NotImplementedError

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(concat_avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(concat_avg_hz_lstm)

    model = Model(inputs=[word_input] + feature_inputs, outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_hrcnn_word_word_model(
        opts,
        model_parameters,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Word level Integration, Word level Features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s, concatenation_mode = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout, opts.concatenation_mode))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)


    if opts.batch_norm:
        bzcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)
        zcnn = TimeDistributed(BatchNormalization(), name='bzcnn')(bzcnn)
    else:
        zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    if opts.feat_embedding > 0:
        feature_input = Input(shape=(N * L,), dtype='int32', name='feature_input')
        fx = Embedding(output_dim=opts.feat_embedding, input_dim=model_parameters['vocab_size'], input_length=N*L, mask_zero=True, name='fx')(feature_input)
        fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
        drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
        resh_fW = Reshape((N, L, opts.feat_embedding), name='resh_fW')(drop_fx)
    elif opts.feat_embedding == 0:
        feature_input = Input(shape=(N, L, model_parameters['vocab_size'],), dtype='float32', name='feature_input')
        resh_fW = feature_input
    else:
        if 'emb_table' in model_parameters:
            feature_input = Input(shape=(model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],),
                                  dtype='int32', name='feature_input')
            fx = Embedding(output_dim=model_parameters['output_dim'],
                           input_dim=model_parameters['vocab_size'],
                           input_length=model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],
                           embeddings_initializer=Constant(model_parameters['emb_table']),
                           trainable=False,
                           mask_zero=True,
                           name='fx')(feature_input)
            fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
            drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
            resh_fW = Reshape((model_parameters['f_max_sentnum'], model_parameters['f_max_sentlen'], model_parameters['output_dim']), name='resh_fW')(drop_fx)
        else:
            feature_input = Input(shape=(N, L, model_parameters['feature_len'],), dtype='float32', name='feature_input')
            resh_fW = feature_input

    if opts.batch_norm:
        bext_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='ext_zcnn')(resh_fW)
        ext_zcnn = TimeDistributed(BatchNormalization(), name='bext_zcnn')(bext_zcnn)
    else:
        ext_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='ext_zcnn')(resh_fW)

    if opts.concatenation_mode == 'concat':
        logger.info(f"Using Simple Concatenation")
        concat_zcnn = K.concatenate([zcnn, ext_zcnn], axis=-1)
    elif opts.concatenation_mode == 'attn':
        logger.info("Using Attention")
        attn_zcnn = Attention(name='attn_concat')([zcnn, ext_zcnn])
        concat_zcnn = K.concatenate([zcnn, attn_zcnn], axis=-1)
    else:
        raise NotImplementedError

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(concat_zcnn)

    if opts.batch_norm:
        bhz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)
        hz_lstm = BatchNormalization(name='bhz_lstm')(bhz_lstm)
    else:
        hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(hz_lstm)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    model = Model(inputs=[word_input, feature_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_hrcnn_sent_word_model(
        opts,
        model_parameters,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Sentence level Integration, Word level Features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s, concatenation_mode = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout, opts.concatenation_mode))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    if opts.batch_norm:
        bzcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)
        zcnn = TimeDistributed(BatchNormalization(), name='bzcnn')(bzcnn)
    else:
        zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)

    if opts.batch_norm:
        bhz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)
        hz_lstm = BatchNormalization(name='bhz_lstm')(bhz_lstm)
    else:
        hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    if opts.feat_embedding > 0:
        feature_input = Input(shape=(N * L,), dtype='int32', name='feature_input')
        fx = Embedding(output_dim=opts.feat_embedding, input_dim=model_parameters['vocab_size'], input_length=N * L,
                       mask_zero=True, name='fx')(feature_input)
        fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
        drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
        resh_fW = Reshape((N, L, opts.feat_embedding), name='resh_fW')(drop_fx)
    elif opts.feat_embedding == 0:
        feature_input = Input(shape=(N, L, model_parameters['vocab_size'],), dtype='float32', name='feature_input')
        resh_fW = feature_input
    else:
        if 'emb_table' in model_parameters:
            feature_input = Input(shape=(model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],),
                                  dtype='int32', name='feature_input')
            fx = Embedding(output_dim=model_parameters['output_dim'],
                           input_dim=model_parameters['vocab_size'],
                           input_length=model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],
                           embeddings_initializer=Constant(model_parameters['emb_table']),
                           trainable=False,
                           mask_zero=True,
                           name='fx')(feature_input)
            fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
            drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
            resh_fW = Reshape((model_parameters['f_max_sentnum'], model_parameters['f_max_sentlen'], model_parameters['output_dim']), name='resh_fW')(drop_fx)
        else:
            feature_input = Input(shape=(N, L, model_parameters['feature_len'],), dtype='float32', name='feature_input')
            resh_fW = feature_input

    if opts.batch_norm:
        bext_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='ext_zcnn')(resh_fW)
        ext_zcnn = TimeDistributed(BatchNormalization(), name='bext_zcnn')(bext_zcnn)
    else:
        ext_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='ext_zcnn')(resh_fW)

    ext_avg_zcnn = TimeDistributed(SelfAttention(), name='ext_avg_zcnn')(ext_zcnn)

    if opts.batch_norm:
        bext_hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='ext_hz_lstm')(ext_avg_zcnn)
        ext_hz_lstm = BatchNormalization(name='bext_hz_lstm')(bext_hz_lstm)
    else:
        ext_hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='ext_hz_lstm')(ext_avg_zcnn)


    if opts.concatenation_mode == 'concat':
        logger.info(f"Using Simple Concatenation")
        concat_hz_lstm = K.concatenate([hz_lstm, ext_hz_lstm], axis=-1)
    elif opts.concatenation_mode == 'attn':
        logger.info("Using Attention")
        attn_hz_lstm = Attention(name='attn_concat')([hz_lstm, ext_hz_lstm])
        concat_hz_lstm = K.concatenate([hz_lstm, attn_hz_lstm], axis=-1)
    else:
        raise NotImplementedError

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(concat_hz_lstm)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    model = Model(inputs=[word_input, feature_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_hrcnn_sent_sent_model(
        opts,
        model_parameters,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Setence level Integration, Sentence level Features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s, concatenation_mode = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout, opts.concatenation_mode))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    if opts.feat_embedding > 0:
        feature_input = Input(shape=(N,), dtype='int32', name='feature_input')
        fx = Embedding(output_dim=opts.feat_embedding, input_dim=model_parameters['vocab_size'], input_length=N,
                       mask_zero=True, name='fx')(feature_input)
        fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
        drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
        tmp_feat_input = drop_fx
    elif opts.feat_embedding == 0:
        feature_input = Input(shape=(N, model_parameters['vocab_size'],), dtype='float32', name='feature_input')
        tmp_feat_input = feature_input
    else:
        if 'emb_table' in model_parameters:
            raise NotImplementedError
        else:
            feature_input = Input(shape=(N, model_parameters['feature_len'],), dtype='float32', name='feature_input')
            tmp_feat_input = feature_input

    ext_hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='ext_hz_lstm')(tmp_feat_input)

    if opts.concatenation_mode == 'concat':
        logger.info(f"Using Simple Concatenation")
        concat_hz_lstm = K.concatenate([hz_lstm, ext_hz_lstm], axis=-1)
    elif opts.concatenation_mode == 'attn':
        logger.info("Using Attention")
        attn_hz_lstm = Attention(name='attn_concat')([hz_lstm, ext_hz_lstm])
        concat_hz_lstm = K.concatenate([hz_lstm, attn_hz_lstm], axis=-1)
    else:
        raise NotImplementedError

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(concat_hz_lstm)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    model = Model(inputs=[word_input, feature_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_hrcnn_doc_word_model(
        opts,
        model_parameters,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Document level Integration, Word level Features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s, concatenation_mode = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout, opts.concatenation_mode))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    if opts.batch_norm:
        bzcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)
        zcnn = TimeDistributed(BatchNormalization(), name='bzcnn')(bzcnn)
    else:
        zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)

    if opts.batch_norm:
        bhz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)
        hz_lstm = BatchNormalization(name='bhz_lstm')(bhz_lstm)
    else:
        hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(hz_lstm)

    if opts.feat_embedding > 0:
        feature_input = Input(shape=(N * L,), dtype='int32', name='feature_input')
        fx = Embedding(output_dim=opts.feat_embedding, input_dim=model_parameters['vocab_size'], input_length=N * L,
                       mask_zero=True, name='fx')(feature_input)
        fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
        drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
        resh_fW = Reshape((N, L, opts.feat_embedding), name='resh_fW')(drop_fx)
    elif opts.feat_embedding == 0:
        feature_input = Input(shape=(N, L, model_parameters['vocab_size'],), dtype='float32', name='feature_input')
        resh_fW = feature_input
    else:
        if 'emb_table' in model_parameters:
            feature_input = Input(shape=(model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],), dtype='int32', name='feature_input')
            fx = Embedding(output_dim=model_parameters['output_dim'],
                           input_dim=model_parameters['vocab_size'],
                           input_length=model_parameters['f_max_sentnum'] * model_parameters['f_max_sentlen'],
                           embeddings_initializer=Constant(model_parameters['emb_table']),
                           trainable=False,
                           mask_zero=True,
                           name='fx')(feature_input)
            fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
            drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
            resh_fW = Reshape((model_parameters['f_max_sentnum'], model_parameters['f_max_sentlen'], model_parameters['output_dim']), name='resh_fW')(drop_fx)
        else:
            feature_input = Input(shape=(N, L, model_parameters['feature_len'],), dtype='float32', name='feature_input')
            resh_fW = feature_input

    if opts.batch_norm:
        bext_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='ext_zcnn')(resh_fW)
        ext_zcnn = TimeDistributed(BatchNormalization(), name='bext_zcnn')(bext_zcnn)
    else:
        ext_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='ext_zcnn')(resh_fW)

    ext_avg_zcnn = TimeDistributed(SelfAttention(), name='ext_avg_zcnn')(ext_zcnn)

    if opts.batch_norm:
        bext_hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='ext_hz_lstm')(ext_avg_zcnn)
        ext_hz_lstm = BatchNormalization(name='bext_hz_lstm')(bext_hz_lstm)
    else:
        ext_hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='ext_hz_lstm')(ext_avg_zcnn)

    ext_avg_hz_lstm = SelfAttention(name='ext_avg_hz_lstm')(ext_hz_lstm)

    if opts.concatenation_mode == 'concat':
        logger.info(f"Using Simple Concatenation")
        concat_hz_lstm = K.concatenate([avg_hz_lstm, ext_avg_hz_lstm], axis=-1)
    elif opts.concatenation_mode == 'attn':
        logger.info("Using Attention")
        attn_hz_lstm = Attention(name='attn_concat')([avg_hz_lstm, ext_avg_hz_lstm])
        concat_hz_lstm = K.concatenate([avg_hz_lstm, attn_hz_lstm], axis=-1)
    else:
        raise NotImplementedError


    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(concat_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(concat_hz_lstm)

    model = Model(inputs=[word_input, feature_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_hrcnn_doc_sent_model(
        opts,
        model_parameters,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Document level Integration, Sentence level Features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s, concatenation_mode = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout, opts.concatenation_mode))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(hz_lstm)

    if opts.feat_embedding > 0:
        feature_input = Input(shape=(N,), dtype='int32', name='feature_input')
        fx = Embedding(output_dim=opts.feat_embedding, input_dim=model_parameters['vocab_size'], input_length=N,
                       mask_zero=True, name='fx')(feature_input)
        fx_maskedout = ZeroMaskedEntries(name='zx_maskedout')(fx)
        drop_fx = Dropout(opts.dropout, name='drop_zx')(fx_maskedout)
        tmp_feat_input = drop_fx
    elif opts.feat_embedding == 0:
        feature_input = Input(shape=(N, model_parameters['vocab_size'],), dtype='float32', name='feature_input')
        tmp_feat_input = feature_input
    else:
        if 'emb_table' in model_parameters:
            raise NotImplementedError
        else:
            feature_input = Input(shape=(N, model_parameters['feature_len'],), dtype='float32', name='feature_input')
            tmp_feat_input = feature_input

    ext_hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='ext_hz_lstm')(tmp_feat_input)

    ext_avg_hz_lstm = SelfAttention(name='ext_avg_hz_lstm')(ext_hz_lstm)

    if opts.concatenation_mode == 'concat':
        logger.info(f"Using Simple Concatenation")
        concat_hz_lstm = K.concatenate([avg_hz_lstm, ext_avg_hz_lstm], axis=-1)
    elif opts.concatenation_mode == 'attn':
        logger.info("Using Attention")
        attn_hz_lstm = Attention(name='attn_concat')([avg_hz_lstm, ext_avg_hz_lstm])
        concat_hz_lstm = K.concatenate([avg_hz_lstm, attn_hz_lstm], axis=-1)
    else:
        raise NotImplementedError


    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(concat_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(concat_hz_lstm)

    model = Model(inputs=[word_input, feature_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_hrcnn_doc_doc_model(
        opts,
        model_parameters,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Document level Integration, Document level Features
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s, concatenation_mode = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout, opts.concatenation_mode))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(hz_lstm)

    feature_input = Input(shape=(model_parameters['feature_len'],), dtype='float32', name='feature_input')

    if opts.concatenation_mode == 'concat':
        logger.info(f"Using Simple Concatenation")
        concat_hz_lstm = K.concatenate([avg_hz_lstm, feature_input], axis=-1)
    elif opts.concatenation_mode == 'attn':
        logger.info("Using Attention")
        attn_hz_lstm = Attention(name='attn_concat')([K.expand_dims(avg_hz_lstm), K.expand_dims(feature_input)])
        concat_hz_lstm = K.concatenate([avg_hz_lstm, K.squeeze(attn_hz_lstm, axis=-1)], axis=-1)
    else:
        raise NotImplementedError


    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(concat_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(concat_hz_lstm)

    model = Model(inputs=[word_input, feature_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model



def build_transformer_hrcnn_model(
        opts,
        transformer_path,
        maxnum=50,
        maxlen=50,
        fine_tune=False,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over transformer on sentence level.
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, transformer = %s, lstm_units = %s" % (
        N, L, opts.transformer_path, opts.lstm_units))
    input_word_ids = Input(shape=(N, L, 1), dtype='int32', name="input_word_ids")
    input_mask = Input(shape=(N, L, 1), dtype='int32', name="input_mask")
    segment_ids = Input(shape=(N, L, 1), dtype='int32', name="segment_ids")

    concat = K.concatenate([input_word_ids, input_mask, segment_ids], axis=-1)

    # logger.info('Use attention-pooling on sentence')
    # avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(transformer_output)

    transformer_layer = AlbertLayer(transformer_path=transformer_path, trainable=fine_tune)
    transformer_output = TimeDistributed(transformer_layer)(concat)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(transformer_output)

    logger.info('Use attention-pooling on text')
    avg_hz_lstm = SelfAttention(name='avg_hz_lstm')(hz_lstm)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model

########################################################################################################################

def build_sthrcnn_model(
        opts,
        topic_len,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Topic attention on sentence level
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    # integrate topic information
    essay_topic_input = Input(shape=(N, topic_len), dtype='float32', name='essay_topic_input')
    # if opts.lstm_units == opts.lda_len:
    #     hz_topic = AdditiveAttention(name='hz_topic')([hz_lstm, essay_topic_input])
    # else:
    essay_topic = Dense(units=opts.lstm_units, name='essay_topic')(essay_topic_input)
    hz_topic = AdditiveAttention(name='hz_topic')([hz_lstm, essay_topic])

    logger.info('Use attention-pooling on text')
    avg_hz_topic = SelfAttention(name='avg_hz_topic')(hz_topic)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_topic)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_topic)

    model = Model(inputs=[word_input, essay_topic_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_ethrcnn_model(
        opts,
        topic_len,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Topic attention on topic level
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)

    # integrate topic information
    # need expand input dims outside
    essay_topic_input = Input(shape=(topic_len, 1), dtype='float32', name='essay_topic_input')
    essay_topic = Dense(units=opts.lstm_units, name='essay_topic')(essay_topic_input)
    hz_topic = AdditiveAttention(name='hz_topic')([hz_lstm, essay_topic])

    logger.info('Use attention-pooling on text')
    avg_hz_topic = SelfAttention(name='avg_hz_topic')(hz_topic)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_topic)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_topic)

    model = Model(inputs=[word_input, essay_topic_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_wtshrcnn_model(
        opts,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        topic_embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Topic attention on topic level
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    topic_x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=topic_embedding_weights, mask_zero=True, name='topic_x')(word_input)
    topic_x_maskedout = ZeroMaskedEntries(name='topic_x_maskedout')(topic_x)
    topic_drop_x = Dropout(opts.dropout, name='topic_drop_x')(topic_x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)
    topic_resh_W = Reshape((N, L, embedd_dim), name='topic_resh_W')(topic_drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)
    topic_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='topic_zcnn')(topic_resh_W)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)
    topic_avg_zcnn = TimeDistributed(SelfAttention(), name='topic_avg_zcnn')(topic_zcnn)

    # hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)
    # topic_hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='topic_hz_lstm')(topic_avg_zcnn)

    hz_topic = AdditiveAttention(name='hz_topic')([avg_zcnn, topic_avg_zcnn])

    final_concat_layer = Concatenate(name='final_concat')([avg_zcnn, hz_topic])

    # logger.info('Use attention-pooling on text')
    # avg_hz_topic = SelfAttention(name='avg_hz_topic')(final_concat_layer)
    html_hz_topic = LSTM(opts.lstm_units, return_sequences=False, name='html_hz_topic')(final_concat_layer)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(html_hz_topic)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(html_hz_topic)

    model = Model(inputs=word_input, outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_wtwhrcnn_model(
        opts,
        vocab_size=0,
        maxnum=50,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        topic_embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN based with ATTN on sentence level, Topic attention on topic level
    N = maxnum
    L = maxlen

    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    topic_x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=topic_embedding_weights, mask_zero=True, name='topic_x')(word_input)
    topic_x_maskedout = ZeroMaskedEntries(name='topic_x_maskedout')(topic_x)
    topic_drop_x = Dropout(opts.dropout, name='topic_drop_x')(topic_x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)
    topic_resh_W = Reshape((N, L, embedd_dim), name='topic_resh_W')(topic_drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)
    topic_zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='topic_zcnn')(topic_resh_W)

    # resh_zcnn = Reshape((N * zcnn.shape[2], opts.nbfilters), name='resh_zcnn')(zcnn)
    # resh_topic_zcnn = Reshape((N * topic_zcnn.shape[2], opts.nbfilters), name='resh_topic_zcnn')(topic_zcnn)

    zcnn_att = Attention(name='zcnn_att')([zcnn, topic_zcnn])
    concat = Concatenate(name='concat')([zcnn, zcnn_att])

    # resh_zcnn_att = Reshape((N, topic_zcnn.shape[2], opts.nbfilters), name='resh_zcnn_att')(zcnn_att)

    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(concat)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)


    logger.info('Use attention-pooling on text')
    avg_hz_topic = SelfAttention(name='avg_hz_topic')(hz_lstm)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_topic)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_topic)

    model = Model(inputs=word_input, outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_wtwcnn_model(
        opts,
        vocab_size=0,
        maxlen=50,
        embedd_dim=50,
        embedding_weights=None,
        topic_embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # CNN based with ATTN on document level, Topic attention on topic level
    L = maxlen

    logger.info("Model parameters: max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(L,), dtype='int32', name='word_input')
    x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    topic_x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=L, weights=topic_embedding_weights, mask_zero=True, name='topic_x')(word_input)
    topic_x_maskedout = ZeroMaskedEntries(name='topic_x_maskedout')(topic_x)
    topic_drop_x = Dropout(opts.dropout, name='topic_drop_x')(topic_x_maskedout)

    zcnn = Conv1D(opts.nbfilters, opts.filter1_len, padding='valid', name='zcnn')(drop_x)
    topic_zcnn = Conv1D(opts.nbfilters, opts.filter1_len, padding='valid', name='topic_zcnn')(topic_drop_x)

    zcnn_att = Attention(name='zcnn_att')([zcnn, topic_zcnn])


    logger.info('Use attention-pooling on text')
    avg_hz_topic = SelfAttention(name='avg_hz_topic')(zcnn_att)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_topic)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_topic)

    model = Model(inputs=word_input, outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model


def build_wtscohrcnn_model(
        opts,
        vocab_size=0,
        topic_vocab_size=0,
        maxnum=50,
        maxlen=50,
        maxnum_topic=50,
        maxlen_topic=50,
        embedd_dim=50,
        embedding_weights=None,
        topic_embedding_weights=None,
        verbose=False,
        init_mean_value=None
):
    # LSTM stacked over CNN with CO-ATTN based on sentence level
    N = maxnum
    L = maxlen

    NT = maxnum_topic
    LT = maxlen_topic


    logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (
        N, L, embedd_dim, opts.nbfilters, opts.filter1_len, opts.dropout))

    word_input = Input(shape=(N * L,), dtype='int32', name='word_input')
    topic_input = Input(shape=(NT * LT,), dtype='int32', name='topic_input')

    emb = Embedding(output_dim=embedd_dim, input_dim=vocab_size, weights=embedding_weights, mask_zero=True, name='cx')
    topic_emb = Embedding(output_dim=embedd_dim, input_dim=topic_vocab_size, input_length=NT*LT, weights=topic_embedding_weights, mask_zero=True, name='topic_x')
    cx = topic_emb(topic_input)
    cx_maskedout = ZeroMaskedEntries(name='cx_maskedout')(cx)
    drop_cx = Dropout(opts.dropout, name='drop_cx')(cx_maskedout)

    resh_C = Reshape((NT, LT, embedd_dim), name='resh_C')(drop_cx)

    czcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='czcnn')(resh_C)

    x = emb(word_input)
    x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
    drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

    resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

    zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

    # pooling mode
    logger.info('Use attention-pooling on sentence')
    avg_zcnn = TimeDistributed(SelfAttention(), name='avg_zcnn')(zcnn)
    avg_czcnn = TimeDistributed(SelfAttention(), name='avg_czcnn')(czcnn)

    hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)
    chz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='chz_lstm')(avg_czcnn)


    logger.info('Use co-attention on text')

    # PART 2:
    # Now we compute a similarity between the passage words and the question words, and
    # normalize the matrix in a couple of different ways for input into some more layers.
    matrix_attention_layer = MatrixAttention(name='essay_context_similarity')
    # Shape: (batch_size, num_passage_words, num_question_words)
    essay_context_similarity = matrix_attention_layer([hz_lstm, chz_lstm])

    # Shape: (batch_size, num_passage_words, num_question_words), normalized over question
    # words for each passage word.
    essay_context_attention = MaskedSoftmax()(essay_context_similarity)
    weighted_sum_layer = WeightedSum(name="essay_context_vectors", use_masking=False)
    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    weighted_hz_lstm = weighted_sum_layer([chz_lstm, essay_context_attention])

    # Min's paper finds, for each document word, the most similar question word to it, and
    # computes a single attention over the whole document using these max similarities.
    # Shape: (batch_size, num_passage_words)
    context_essay_similarity = Max(axis=-1)(essay_context_similarity)
    # Shape: (batch_size, num_passage_words)
    context_essay_attention = MaskedSoftmax()(context_essay_similarity)
    # Shape: (batch_size, embedding_dim * 2)
    weighted_sum_layer = WeightedSum(name="context_essay_vector", use_masking=False)
    context_essay_vector = weighted_sum_layer([hz_lstm, context_essay_attention])

    # Then he repeats this question/passage vector for every word in the passage, and uses it
    # as an additional input to the hidden layers above.
    repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
    # Shape: (batch_size, num_passage_words, embedding_dim * 2)
    tiled_context_essay_vector = repeat_layer([context_essay_vector, hz_lstm])

    complex_concat_layer = ComplexConcat(combination='1,2,1*2,1*3', name='final_merged_passage')
    final_merged_passage = complex_concat_layer([hz_lstm,
                                                 weighted_hz_lstm,
                                                 tiled_context_essay_vector])

    avg_hz_lstm = LSTM(opts.lstm_units, return_sequences=False, name='avg_hz_lstm')(final_merged_passage)

    if opts.l2_value:
        logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
        y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
    else:
        y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

    model = Model(inputs=[word_input, topic_input], outputs=y)

    if opts.init_bias and init_mean_value:
        logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
        bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
        model.layers[-1].b.set_value(bias_value)

    if verbose:
        model.summary()

    compile_model(model, opts)

    return model