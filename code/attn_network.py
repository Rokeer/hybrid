# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/11/19 1:21 PM

import argparse
import time
import datetime
import numpy as np
import tensorflow as tf
from utils import get_logger
import data_prepare

from networks.network_models import build_hrcnn_model, build_shrcnn_model
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

    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of texts in each batch')
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
    parser.add_argument('--fold', type=str, help='fold number', default='10')
    parser.add_argument('--data_base', type=str, help='data path base folder', default='data/data_with_eRevise_18fall/fold_')
    parser.add_argument('--prompt_id', type=int, default=9, help='prompt id of essay set')
    parser.add_argument('--init_bias', action='store_true',
                        help='init the last layer bias with average score of training data')
    parser.add_argument('--mode', type=str, choices=['att', 'co'],
                        default='co', help='attention-pooling, or co-attention pooling')


    # parser.add_argument('--lda_len', type=int, default=100, help='Num of topic of LDA model')
    # parser.add_argument('--lda_file', type=str, default='lda/sent_50', help='Pretrained LDA model path')

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

    datapaths = [
        args.data_base + fold + '/train.tsv',
        args.data_base + fold + '/dev.tsv',
        args.data_base + fold + '/test.tsv'
    ]
    embedding_path = args.embedding_dict
    embedding = args.embedding
    emb_dim = args.embedding_dim
    prompt_id = args.prompt_id

    mode = args.mode
    need_context = mode in ['co']
    sentence_level = mode not in ['word_topic_no_hire_word']

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

    if mode == 'att':
        model = build_hrcnn_model(args, vocab_size, max_sentnum, max_sentlen, emb_dim, emb_table, True, init_mean_value)
        x_train = X_train
        y_train = Y_train
        x_dev = X_dev
        y_dev = Y_dev
        x_test = X_test
        y_test = Y_test
    elif mode == 'co':
        model = build_shrcnn_model(args, vocab_size, max_sentnum, max_sentlen, context_num, context_len, emb_dim,
                                   emb_table, True, init_mean_value)
        x_train = [X_train, train_context]
        y_train = Y_train
        x_dev = [X_dev, dev_context]
        y_dev = Y_dev
        x_test = [X_test, test_context]
        y_test = Y_test
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
        str(args)
    )

    # Initial evaluation
    logger.info("Initial evaluation: ")
    evl.evaluate(model, -1, print_info=True)
    logger.info("Train model")
    for ii in range(num_epochs):
        logger.info('Epoch %s/%s' % (str(ii + 1), num_epochs))
        start_time = time.time()
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time.time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        evl.evaluate(model, ii + 1, print_info=True)

    evl.print_final_info()


if __name__ == '__main__':
    main()
