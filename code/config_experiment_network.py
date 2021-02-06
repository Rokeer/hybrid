# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   5/26/20 3:10 PM

import argparse
import time
import datetime
import numpy as np
import tensorflow as tf
from utils import get_logger
import data_prepare
import feature_extractor as fe
import os
import pickle
import json

from tagsets import discourse_tagset, modal_tagset, pos_tag_tagset, wn_pos_tag_tagset, argumentation_tagset, sent_function_label_tagset

from networks.network_models import build_hrcnn_model, build_shrcnn_model
# from networks.experiment_models import build_hrcnn_doc_doc_model, build_hrcnn_doc_sent_model, build_hrcnn_doc_word_model
# from networks.experiment_models import build_hrcnn_sent_sent_model, build_hrcnn_sent_word_model
# from networks.experiment_models import build_hrcnn_word_word_model
from networks.experiment_models import build_hrcnn_combined_model
from evaluator import Evaluator


logger = get_logger("Main")
np.random.seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # logger.info(str(len(gpus)) + " Physical GPUs, " + str(len(logical_gpus)) + " Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(e)


def main():
    parser = argparse.ArgumentParser(description="sentence Hi_CNN_LSTM model")
    parser.add_argument('--embedding', type=str, default='glove',
                        help='Word embedding type, glove, word2vec, senna or random')
    parser.add_argument('--embedding_dict', type=str, default='glove/glove.6B.50d.txt',
                        help='Pretrained embedding path')
    parser.add_argument('--embedding_dim', type=int, default=50,
                        help='Only useful when embedding is randomly initialised')

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of texts in each batch')
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000,
                        help="Vocab size (default=4000)")

    parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')
    parser.add_argument('--filter1_len', type=int, default=5, help='filter length in 1st conv layer')
    parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

    parser.add_argument('--optimizer', choices=['sgd', 'adagrad', 'rmsprop'], help='Optimizer', default='rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
    parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory', default='checkpoints')

    # parser.add_argument('--train', type=str, help='train file', default='data/fold_0/train.tsv')
    # parser.add_argument('--dev', type=str, help='dev file', default='data/fold_0/dev.tsv')
    # parser.add_argument('--test', type=str, help='test file', default='data/fold_0/test.tsv')
    parser.add_argument('--fold', type=str, help='fold number', default='0')
    parser.add_argument('--data_base', type=str, help='data path base folder', default='data/fold_')
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')
    parser.add_argument('--init_bias', action='store_true',
                        help='init the last layer bias with average score of training data')
    parser.add_argument('--mode', type=str, choices=['att', 'co', 'experiment'],
                        default='experiment', help='attention-pooling, or co-attention pooling')
    parser.add_argument('--concatenation_mode', type=str, choices=['concat', 'attn', 'co', 'config'], default='attn',
                        help='simple concatenation or attention concatenation')
    parser.add_argument('--config', type=str, help='experiment config file', default='experiment_config')


    feature_choices = [
        'sentiment_word',
        'cat_dis_word',
        'modal_word',
        'pos_tag_seq_word',
        'wn_pos_tag_seq_word',
        'lda_word',
        'lda_sent',
        'lda_doc',
        'readability_sent',
        'readability_doc',
        'argumentation_word',
        'discourse_func_sent',
        'discourse_func_vec_sent',
        'word_count_sent',
        'word_count_doc',
        'rta_doc'
    ]


    parser.add_argument('--stanford_path', type=str, default='/Users/colin/Documents/Work/libs/stanford-corenlp-full-2018-10-05/', help='path of stanford corenlp full')

    parser.add_argument('--topic_model_mode', type=str, default='lda', choices=['lda', 'mallet'], help='name of topic model')
    parser.add_argument('--lda_len', type=int, default=50, help='length of lda model')

    parser.add_argument('--cross_domain', action='store_true', help='is this a cross domain experiments?')
    parser.add_argument('--source_prompt', type=int, default=1, help='source prompt for cross domain experiments')
    parser.add_argument('--target_prompt', type=int, default=2, help='target prompt for cross domain experiments')
    parser.add_argument('--sample_size', type=int, default=10, help='samples to add from target prompt')
    parser.add_argument('--dev_size', type=int, default=0, help='size of dev, -1: use source dev, 0: use all target dev, positive num: use num of target dev')

    parser.add_argument('--need_context', action='store_true', help='if it need to read context')
    parser.add_argument('--use_co_model', action='store_true', help='if use lstm to get final output')
    args = parser.parse_args()



    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_path
    num_epochs = args.num_epochs
    mode = args.mode
    prompt_id = args.prompt_id
    fold = args.fold

    date = datetime.datetime.now()
    ts = date.strftime("%Y.%m.%d.%H.%M.%S")
    modelname = "%s.%s.prompt%s.fold%s.bs%s" % (ts, mode, prompt_id, fold, batch_size)

    cross_domain = args.cross_domain
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    sample_size = args.sample_size
    dev_size = args.dev_size

    if cross_domain:
        logger.warning('Since cross_domain is True, prompt_id will be set to target_prompt!')
        # reset prompt_id to target_prompt
        prompt_id = target_prompt
        modelname = "%s.%s.source_prompt%s.target_prompt%s.sample_size%s.dev_size%s.fold%s.bs%s" % (ts, mode, source_prompt, target_prompt, sample_size, dev_size, fold, batch_size)

    config_file = 'experiment_configs/' + args.config + '.json'
    if mode == "experiment":
        if os.path.exists(config_file):
            with open(config_file) as f:
                json_obj = json.load(f)
                if 'experiments' in json_obj:
                    configs = json_obj['experiments']
                elif 'pd_exp' in json_obj:
                    configs = json_obj['pd_exp']['p' + str(prompt_id)]
                    if cross_domain:
                        configs = json_obj['pd_exp']['p' + str(source_prompt)]
                else:
                    configs = []
                experiment_name = args.config
        else:
            configs = []
            experiment_name = 'config_not_found'
    else:
        configs = []
        experiment_name = mode

    if cross_domain:
        experiment_name = experiment_name + '_' + str(source_prompt) + '_' + str(target_prompt) + '_' + str(sample_size) + '_' + str(dev_size)

    datapaths = [
        args.data_base + fold + '/train.tsv',
        args.data_base + fold + '/dev.tsv',
        args.data_base + fold + '/test.tsv'
    ]

    if cross_domain:
        datapaths.append(args.data_base + fold + '/shuffled_train.json')

    embedding_path = args.embedding_dict
    embedding = args.embedding
    emb_dim = args.embedding_dim

    stanford_path = args.stanford_path

    topic_model_mode = args.topic_model_mode
    lda_len = args.lda_len

    need_context = (mode in ['co']) or args.need_context
    sentence_level = mode not in ['word_topic_no_hire_word']

    checkpoint_dir = checkpoint_dir + '/' + experiment_name

    if cross_domain:
        (X_train, Y_train, mask_train, train_context, text_train, train_ids), \
        (X_dev, Y_dev, mask_dev, dev_context, text_dev, dev_ids), \
        (X_test, Y_test, mask_test, test_context, text_test, test_ids), \
        vocab, vocab_size, emb_table, overall_maxlen, overall_maxnum, init_mean_value, context_len, context_num = \
            data_prepare.prepare_cross_sentence_data(
                datapaths,
                embedding_path,
                embedding,
                emb_dim,
                source_prompt,
                target_prompt,
                sample_size,
                dev_size,
                args.vocab_size,
                tokenize_text=True,
                to_lower=True,
                replace_num=True,
                vocab_path=None,
                score_index=6,
                need_context=need_context,
                sentence_level=sentence_level
            )
    else:
        (X_train, Y_train, mask_train, train_context, text_train, train_ids), \
        (X_dev, Y_dev, mask_dev, dev_context, text_dev, dev_ids), \
        (X_test, Y_test, mask_test, test_context, text_test, test_ids), \
        vocab, vocab_size, emb_table, overall_maxlen, overall_maxnum, init_mean_value, context_len, context_num = \
            data_prepare.prepare_sentence_data(
                datapaths,
                embedding_path,
                embedding,
                emb_dim,
                prompt_id,
                args.vocab_size,
                tokenize_text=True,
                to_lower=True,
                replace_num=True,
                vocab_path=None,
                score_index=6,
                need_context=need_context,
                sentence_level=sentence_level
            )

    if emb_table is not None:
        emb_dim = emb_table.shape[1]
        emb_table = [emb_table]

    max_sentnum = overall_maxnum
    max_sentlen = overall_maxlen

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1] * X_dev.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    train_context = train_context.reshape((train_context.shape[0], train_context.shape[1] * train_context.shape[2]))
    dev_context = dev_context.reshape((dev_context.shape[0], dev_context.shape[1] * dev_context.shape[2]))
    test_context = test_context.reshape((test_context.shape[0], test_context.shape[1] * test_context.shape[2]))

    logger.info("X_train shape: %s" % str(X_train.shape))
    logger.info("X_dev shape: %s" % str(X_dev.shape))
    logger.info("X_test shape: %s" % str(X_test.shape))

    # with open('tmp/X_train.pkl', 'wb') as f:
    #     pickle.dump(X_train, f)
    # with open('tmp/X_dev.pkl', 'wb') as f:
    #     pickle.dump(X_dev, f)
    # with open('tmp/X_test.pkl', 'wb') as f:
    #     pickle.dump(X_test, f)
    # with open('tmp/Y_train.pkl', 'wb') as f:
    #     pickle.dump(Y_train, f)
    # with open('tmp/Y_dev.pkl', 'wb') as f:
    #     pickle.dump(Y_dev, f)
    # with open('tmp/Y_test.pkl', 'wb') as f:
    #     pickle.dump(Y_test, f)
    # with open('tmp/text_train.pkl', 'wb') as f:
    #     pickle.dump(text_train, f)
    # with open('tmp/text_dev.pkl', 'wb') as f:
    #     pickle.dump(text_dev, f)
    # with open('tmp/text_test.pkl', 'wb') as f:
    #     pickle.dump(text_test, f)
    # with open('tmp/train_ids.pkl', 'wb') as f:
    #     pickle.dump(train_ids, f)
    # with open('tmp/dev_ids.pkl', 'wb') as f:
    #     pickle.dump(dev_ids, f)
    # with open('tmp/test_ids.pkl', 'wb') as f:
    #     pickle.dump(test_ids, f)
    # with open('tmp/vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)

    # with open('tmp/X_train.pkl', 'rb') as f:
    #     X_train = pickle.load(f)
    # with open('tmp/X_dev.pkl', 'rb') as f:
    #     X_dev = pickle.load(f)
    # with open('tmp/X_test.pkl', 'rb') as f:
    #     X_test = pickle.load(f)
    # with open('tmp/Y_train.pkl', 'rb') as f:
    #     Y_train = pickle.load(f)
    # with open('tmp/Y_dev.pkl', 'rb') as f:
    #     Y_dev = pickle.load(f)
    # with open('tmp/Y_test.pkl', 'rb') as f:
    #     Y_test = pickle.load(f)
    # with open('tmp/text_train.pkl', 'rb') as f:
    #     text_train = pickle.load(f)
    # with open('tmp/text_dev.pkl', 'rb') as f:
    #     text_dev = pickle.load(f)
    # with open('tmp/text_test.pkl', 'rb') as f:
    #     text_test = pickle.load(f)
    # with open('tmp/train_ids.pkl', 'rb') as f:
    #     train_ids = pickle.load(f)
    # with open('tmp/dev_ids.pkl', 'rb') as f:
    #     dev_ids = pickle.load(f)
    # with open('tmp/test_ids.pkl', 'rb') as f:
    #     test_ids = pickle.load(f)
    # with open('tmp/vocab.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    # vocab_size = 4000
    # max_sentnum = 88
    # max_sentlen = 50
    # emb_table = None
    # init_mean_value = 0.65532714

    x_train = [X_train]
    x_dev = [X_dev]
    x_test = [X_test]
    y_train = Y_train
    y_dev = Y_dev
    y_test = Y_test
    features = []

    for config in configs:
        feature = config['feature']
        model_parameters = {}
        f_train, f_dev, f_test = None, None, None
        if feature == 'cat_dis_word':
            logger.info('Handcraft feature: Word Category Feature - Discourse Connectives')

            f_train, f_dev, f_test = fe.cat_dis_word(
                stanford_path,
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                max_sentlen,
                cross_domain,
                source_prompt
            )

            if config['embedding_size'] > 0:
                f_train = f_train.reshape((f_train.shape[0], f_train.shape[1] * f_train.shape[2]))
                f_dev = f_dev.reshape((f_dev.shape[0], f_dev.shape[1] * f_dev.shape[2]))
                f_test = f_test.reshape((f_test.shape[0], f_test.shape[1] * f_test.shape[2]))
                model_parameters['vocab_size'] = len(discourse_tagset) + 1
                features.append(feature + '_emb' + str(config['embedding_size']))
            else:
                logger.info('Reset feat_embedding to 0')
                config['embedding_size'] = 0
                f_train = tf.keras.utils.to_categorical(f_train, len(discourse_tagset) + 1)[:, :, :, 1:]
                f_dev = tf.keras.utils.to_categorical(f_dev, len(discourse_tagset) + 1)[:, :, :, 1:]
                f_test = tf.keras.utils.to_categorical(f_test, len(discourse_tagset) + 1)[:, :, :, 1:]
                model_parameters['vocab_size'] = len(discourse_tagset)
                features.append(feature + '_one_hot')

            model_parameters['feature_len'] = 1
        if feature == 'sentiment_word':
            logger.info('Handcraft feature: Word Category Feature - Sentiment Words')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature)

            f_train, f_dev, f_test = fe.sentiment_word(
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                max_sentlen,
                cross_domain,
                source_prompt
            )

            model_parameters['feature_len'] = 3

        if feature == 'modal_word':
            logger.info('Handcraft feature: Word Category Feature - Modal Verbs')

            f_train, f_dev, f_test = fe.modal_word(
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                max_sentlen,
                cross_domain,
                source_prompt
            )

            if config['embedding_size'] > 0:
                f_train = f_train.reshape((f_train.shape[0], f_train.shape[1] * f_train.shape[2]))
                f_dev = f_dev.reshape((f_dev.shape[0], f_dev.shape[1] * f_dev.shape[2]))
                f_test = f_test.reshape((f_test.shape[0], f_test.shape[1] * f_test.shape[2]))
                model_parameters['vocab_size'] = len(modal_tagset) + 1
                features.append(feature + '_emb' + str(config['embedding_size']))
            else:
                logger.info('Reset feat_embedding to 0')
                config['embedding_size'] = 0
                f_train = tf.keras.utils.to_categorical(f_train, len(modal_tagset) + 1)[:, :, :, 1:]
                f_dev = tf.keras.utils.to_categorical(f_dev, len(modal_tagset) + 1)[:, :, :, 1:]
                f_test = tf.keras.utils.to_categorical(f_test, len(modal_tagset) + 1)[:, :, :, 1:]
                model_parameters['vocab_size'] = len(modal_tagset)
                features.append(feature + '_one_hot')

            model_parameters['feature_len'] = 1

        if feature == 'pos_tag_seq_word':
            logger.info('Handcraft feature: Syntactic Feature - Pos Tag Sequence')

            f_train, f_dev, f_test = fe.pos_tag_seq_word(
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                max_sentlen,
                True,
                cross_domain,
                source_prompt
            )

            if config['embedding_size'] > 0:
                f_train = f_train.reshape((f_train.shape[0], f_train.shape[1] * f_train.shape[2]))
                f_dev = f_dev.reshape((f_dev.shape[0], f_dev.shape[1] * f_dev.shape[2]))
                f_test = f_test.reshape((f_test.shape[0], f_test.shape[1] * f_test.shape[2]))
                model_parameters['vocab_size'] = len(pos_tag_tagset) + 1
                features.append(feature + '_emb' + str(config['embedding_size']))
            else:
                logger.info('Reset feat_embedding to 0')
                config['embedding_size'] = 0
                f_train = tf.keras.utils.to_categorical(f_train, len(pos_tag_tagset) + 1)[:, :, :, 1:]
                f_dev = tf.keras.utils.to_categorical(f_dev, len(pos_tag_tagset) + 1)[:, :, :, 1:]
                f_test = tf.keras.utils.to_categorical(f_test, len(pos_tag_tagset) + 1)[:, :, :, 1:]
                model_parameters['vocab_size'] = len(pos_tag_tagset)
                features.append(feature + '_one_hot')

            model_parameters['feature_len'] = 1

        if feature == 'wn_pos_tag_seq_word':
            logger.info('Handcraft feature: Syntactic Feature - WordNet Pos Tag Sequence')

            f_train, f_dev, f_test = fe.pos_tag_seq_word(
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                max_sentlen,
                False,
                cross_domain,
                source_prompt
            )

            if config['embedding_size'] > 0:
                f_train = f_train.reshape((f_train.shape[0], f_train.shape[1] * f_train.shape[2]))
                f_dev = f_dev.reshape((f_dev.shape[0], f_dev.shape[1] * f_dev.shape[2]))
                f_test = f_test.reshape((f_test.shape[0], f_test.shape[1] * f_test.shape[2]))
                model_parameters['vocab_size'] = len(wn_pos_tag_tagset) + 1
                features.append(feature + '_emb' + str(config['embedding_size']))
            else:
                logger.info('Reset feat_embedding to 0')
                config['embedding_size'] = 0
                f_train = tf.keras.utils.to_categorical(f_train, len(wn_pos_tag_tagset) + 1)[:, :, :, 1:]
                f_dev = tf.keras.utils.to_categorical(f_dev, len(wn_pos_tag_tagset) + 1)[:, :, :, 1:]
                f_test = tf.keras.utils.to_categorical(f_test, len(wn_pos_tag_tagset) + 1)[:, :, :, 1:]
                model_parameters['vocab_size'] = len(wn_pos_tag_tagset)
                features.append(feature + '_one_hot')

            model_parameters['feature_len'] = 1

        if feature == 'lda_word':
            logger.info('Handcraft feature: Prompt-relevant Feature - LDA')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature + '_' + topic_model_mode + '_' + str(lda_len))

            f_train, f_dev, f_test, f_emb_table, f_max_sentnum, f_max_sentlen = fe.lda_word(
                text_train,
                text_dev,
                text_test,
                prompt_id,
                fold,
                topic_model_mode,
                lda_len,
                cross_domain,
                source_prompt,
                sample_size,
                dev_size
            )

            f_train = f_train.reshape((f_train.shape[0], f_train.shape[1] * f_train.shape[2]))
            f_dev = f_dev.reshape((f_dev.shape[0], f_dev.shape[1] * f_dev.shape[2]))
            f_test = f_test.reshape((f_test.shape[0], f_test.shape[1] * f_test.shape[2]))

            model_parameters['vocab_size'] = f_emb_table[0].shape[0]
            model_parameters['output_dim'] = f_emb_table[0].shape[1]
            model_parameters['emb_table'] = f_emb_table
            model_parameters['fine_tune'] = False
            model_parameters['f_max_sentnum'] = f_max_sentnum
            model_parameters['f_max_sentlen'] = f_max_sentlen

            model_parameters['feature_len'] = 1

        if feature == 'argumentation_word':
            logger.info('Handcraft feature: Argumentation Feature- Argumentation Components Sequence')

            f_train, f_dev, f_test = fe.argumentation_word(
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                max_sentlen,
                cross_domain,
                source_prompt
            )

            if config['embedding_size'] > 0:
                f_train = f_train.reshape((f_train.shape[0], f_train.shape[1] * f_train.shape[2]))
                f_dev = f_dev.reshape((f_dev.shape[0], f_dev.shape[1] * f_dev.shape[2]))
                f_test = f_test.reshape((f_test.shape[0], f_test.shape[1] * f_test.shape[2]))
                model_parameters['vocab_size'] = len(argumentation_tagset) + 1
                features.append(feature + '_emb' + str(config['embedding_size']))
            else:
                logger.info('Reset feat_embedding to 0')
                config['embedding_size'] = 0
                f_train = tf.keras.utils.to_categorical(f_train, len(argumentation_tagset) + 1)[:, :, :, 1:]
                f_dev = tf.keras.utils.to_categorical(f_dev, len(argumentation_tagset) + 1)[:, :, :, 1:]
                f_test = tf.keras.utils.to_categorical(f_test, len(argumentation_tagset) + 1)[:, :, :, 1:]
                model_parameters['vocab_size'] = len(argumentation_tagset)
                features.append(feature + '_one_hot')

            model_parameters['feature_len'] = 1

        if feature == 'lda_sent':
            logger.info('Handcraft feature: Prompt-relevant Feature - LDA')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature + '_' + topic_model_mode + '_' + str(lda_len))

            f_train, f_dev, f_test = fe.lda_sent(
                text_train,
                text_dev,
                text_test,
                prompt_id,
                fold,
                topic_model_mode,
                lda_len,
                cross_domain,
                source_prompt,
                sample_size,
                dev_size
            )

            model_parameters['feature_len'] = lda_len

        if feature == 'readability_sent':
            logger.info('Handcraft feature: Readability Feature')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature)

            f_train, f_dev, f_test = fe.readability_sent(
                text_train,
                text_dev,
                text_test,
                max_sentnum
            )

            model_parameters['feature_len'] = 1

        if feature == 'discourse_func_sent':
            logger.info('Handcraft feature: Discourse Feature - Discourse Function Label')

            f_train, f_dev, f_test = fe.discourse_func_sent(
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                True,
                cross_domain,
                source_prompt,
                sample_size
            )

            if config['embedding_size'] > 0:
                f_train = f_train.squeeze()
                f_dev = f_dev.squeeze()
                f_test = f_test.squeeze()
                model_parameters['vocab_size'] = len(sent_function_label_tagset) + 1
                features.append(feature + '_emb' + str(config['embedding_size']))
            else:
                logger.info('Reset feat_embedding to 0')
                config['embedding_size'] = 0
                f_train = tf.keras.utils.to_categorical(f_train, len(sent_function_label_tagset) + 1)[:, :, 1:]
                f_dev = tf.keras.utils.to_categorical(f_dev, len(sent_function_label_tagset) + 1)[:, :, 1:]
                f_test = tf.keras.utils.to_categorical(f_test, len(sent_function_label_tagset) + 1)[:, :, 1:]
                model_parameters['vocab_size'] = len(sent_function_label_tagset)
                features.append(feature + '_one_hot')

            model_parameters['feature_len'] = 1

        if feature == 'discourse_func_vec_sent':
            logger.info('Handcraft feature: Discourse Feature - Discourse Function Label Vector')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature)

            f_train, f_dev, f_test = fe.discourse_func_sent(
                text_train,
                text_dev,
                text_test,
                train_ids,
                dev_ids,
                test_ids,
                prompt_id,
                max_sentnum,
                False,
                cross_domain,
                source_prompt,
                sample_size
            )

            model_parameters['feature_len'] = len(sent_function_label_tagset)

        if feature == 'word_count_sent':
            logger.info('Handcraft feature: Word Count Feature - Sentence Word Count')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature)

            f_train, f_dev, f_test = fe.word_count_sent(
                text_train,
                text_dev,
                text_test,
                max_sentnum
            )

            model_parameters['feature_len'] = 1

        if feature == 'lda_doc':
            logger.info('Handcraft feature: Prompt-relevant Feature - LDA')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature + '_' + topic_model_mode + '_' + str(lda_len))

            f_train, f_dev, f_test = fe.lda_doc(
                text_train,
                text_dev,
                text_test,
                prompt_id,
                fold,
                topic_model_mode,
                lda_len,
                cross_domain,
                source_prompt,
                sample_size,
                dev_size
            )

            model_parameters['feature_len'] = lda_len

        if feature == 'readability_doc':
            logger.info('Handcraft feature: Readability Feature')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature)

            f_train, f_dev, f_test = fe.readability_doc(
                text_train,
                text_dev,
                text_test
            )

            model_parameters['feature_len'] = 1

        if feature == 'word_count_doc':
            logger.info('Handcraft feature: Word Count Feature - Doc Word Count')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature)

            f_train, f_dev, f_test = fe.word_count_doc(
                text_train,
                text_dev,
                text_test
            )

            model_parameters['feature_len'] = 1

        if feature == 'rta_doc':
            logger.info('Handcraft feature: Word Count Feature - Doc Word Count')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1
            features.append(feature)

            if prompt_id < 9:
                logger.info('rta_doc feature only works for RTA dataset (current: ' + str(prompt_id) + ', skip...')

            f_train, f_dev, f_test = fe.rta_doc(
                train_ids,
                dev_ids,
                test_ids,
                prompt_id
            )

            # hard code here. although I can use f_train.shape[1], but this is somehow a double verification.
            model_parameters['feature_len'] = 11
        if feature == 'source_text_word':
            logger.info('Handcraft feature: Source Text Feature - Source Text')
            logger.info('Reset feat_embedding to -1')
            config['embedding_size'] = -1

            f_train = train_context
            f_dev = dev_context
            f_test = test_context

            model_parameters['use_same_emb'] = 1
            model_parameters['f_max_sentlen'] = context_len
            model_parameters['f_max_sentnum'] = context_num
            model_parameters['output_dim'] = emb_dim


        x_train.append(f_train)
        x_dev.append(f_dev)
        x_test.append(f_test)
        config['model_parameters'] = model_parameters

    # with open(fold + '_train.pkl', 'wb') as pickle_file:
    #     pickle.dump(x_train, pickle_file)
    # with open(fold + '_dev.pkl', 'wb') as pickle_file:
    #     pickle.dump(x_dev, pickle_file)
    # with open(fold + '_test.pkl', 'wb') as pickle_file:
    #     pickle.dump(x_train, pickle_file)
    # exit()

    # create dir for this experiment
    # here is a redundant check
    if not os.path.exists(checkpoint_dir):
        try:
            os.mkdir(checkpoint_dir)
        except FileExistsError:
            logger.info("Directory already exists")
    else:
        logger.info("Directory already exists")

    if mode == 'att':
        model = build_hrcnn_model(args, vocab_size, max_sentnum, max_sentlen, emb_dim, emb_table, True, init_mean_value)
        x_train = X_train
        y_train = Y_train
        x_dev = X_dev
        y_dev = Y_dev
        x_test = X_test
        y_test = Y_test
        args.feature = 'att'
    elif mode == 'co':
        model = build_shrcnn_model(args, vocab_size, max_sentnum, max_sentlen, context_num, context_len, emb_dim,
                                   emb_table, True, init_mean_value)
        x_train = [X_train, train_context]
        y_train = Y_train
        x_dev = [X_dev, dev_context]
        y_dev = Y_dev
        x_test = [X_test, test_context]
        y_test = Y_test
        args.feature = 'co'
    elif mode == 'experiment':
        model = build_hrcnn_combined_model(args, configs, vocab_size, max_sentnum, max_sentlen, emb_dim, emb_table, True, init_mean_value)
        args.feature = '-'.join(features)
    else:
        raise NotImplementedError

    evl = Evaluator(
        prompt_id,
        checkpoint_dir,
        modelname,
        x_train,
        x_dev,
        x_test,
        y_train,
        y_dev,
        y_test,
        ts,
        mode,
        fold,
        batch_size,
        args
    )

    # Initial evaluation
    logger.info("Initial evaluation: ")
    # if cross_domain:
    #     evl.evaluate_by_loss(model, -1, print_info=True)
    # else:
    #     evl.evaluate(model, -1, print_info=True)
    evl.evaluate(model, -1, print_info=True)
    logger.info("Train model")
    for ii in range(num_epochs):
        logger.info('Epoch %s/%s' % (str(ii + 1), num_epochs))
        start_time = time.time()
        if cross_domain:
            history = model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        else:
            history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        print(history.history)
        tt_time = time.time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        # if cross_domain:
        #     evl.evaluate_by_loss(model, ii + 1, current_loss=history.history['val_loss'][0], print_info=True)
        # else:
        #     evl.evaluate(model, ii + 1, print_info=True)
        evl.evaluate(model, ii + 1, print_info=True)

    evl.print_final_info()


if __name__ == '__main__':
    main()
