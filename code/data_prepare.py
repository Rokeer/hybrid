# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/11/19 12:57 PM

import reader
import utils
import os
import pickle
import numpy as np
from features.lda.topic_model import TopicModel

logger = utils.get_logger("Prepare data ...")


def prepare_transformer_data(
        datapaths,
        prompt_id,
        transformer_path
):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"


    (train_x, train_y, train_text), \
    (dev_x, dev_y, dev_text), \
    (test_x, test_y, test_text), \
    overall_maxlen, overall_maxnum = reader.get_transformer_data(datapaths, prompt_id, transformer_path)

    train_input, train_segment, train_masks, y_train = \
        utils.convert_bert_sequence(train_x, train_y, overall_maxnum, overall_maxlen)
    dev_input, dev_segment, dev_masks, y_dev = \
        utils.convert_bert_sequence(dev_x, dev_y, overall_maxnum, overall_maxlen)
    test_input, test_segment, test_masks, y_test = \
        utils.convert_bert_sequence(test_x, test_y, overall_maxnum, overall_maxlen)

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)

    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    Y_train = utils.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = utils.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = utils.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = utils.get_model_friendly_scores(train_mean, prompt_id)
    scaled_dev_mean = utils.get_model_friendly_scores(dev_mean, prompt_id)
    scaled_test_mean = utils.get_model_friendly_scores(test_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(train_input.shape))
    logger.info('  dev X shape:   ' + str(dev_input.shape))
    logger.info('  test X shape:  ' + str(test_input.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))
    logger.info('  dev_y mean: %s, stdev: %s, dev_y mean after scaling: %s' %
                (str(dev_mean), str(dev_std), str(scaled_dev_mean)))
    logger.info('  test_y mean: %s, stdev: %s, test_y mean after scaling: %s' %
                (str(test_mean), str(test_std), str(scaled_test_mean)))

    return (train_input, train_segment, train_masks, Y_train, train_text), \
           (dev_input, dev_segment, dev_masks, Y_dev, dev_text), \
           (test_input, test_segment, test_masks, Y_test, test_text), \
           overall_maxlen, overall_maxnum, scaled_train_mean


def prepare_cross_sentence_data(
        datapaths,
        embedding_path=None,
        embedding='word2vec',
        emb_dim=100,
        source_prompt=1,
        target_prompt=2,
        sample_size=10,
        dev_size=0,
        vocab_size=0,
        tokenize_text=True,
        to_lower=True,
        replace_num=True,
        vocab_path=None,
        score_index=6,
        need_context=True,
        sentence_level=True
):
    assert len(datapaths) == 4, "data paths should include train, dev, test, shuffled_id path"
    (train_x, train_y, train_ids, train_text), \
    (dev_x, dev_y, dev_ids, dev_text), \
    (test_x, test_y, test_ids, test_text), \
    vocab, overall_maxlen, overall_maxnum = \
        reader.get_cross_data(
            datapaths,
            source_prompt,
            target_prompt,
            sample_size,
            dev_size,
            vocab_size,
            tokenize_text,
            to_lower,
            replace_num,
            vocab_path,
            score_index,
            sentence_level
        )

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overall_maxnum, overall_maxlen,
                                                                    post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overall_maxnum, overall_maxlen,
                                                              post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overall_maxnum, overall_maxlen,
                                                                 post_padding=True)

    if need_context:
        raise NotImplementedError
    else:
        # Dummy context
        context = [[0]]
        context_len = 1
        context_num = 1
    train_context = [context] * len(train_x)
    dev_context = [context] * len(dev_x)
    test_context = [context] * len(test_x)

    train_context, _, _ = utils.padding_sentence_sequences(train_context, train_y, context_num, context_len,
                                                           post_padding=True)
    dev_context, _, _ = utils.padding_sentence_sequences(dev_context, dev_y, context_num, context_len,
                                                         post_padding=True)
    test_context, _, _ = utils.padding_sentence_sequences(test_context, test_y, context_num, context_len,
                                                          post_padding=True)

    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)

    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    source_y = utils.get_model_friendly_scores(y_train[0: len(y_train) - sample_size], source_prompt)
    target_y = utils.get_model_friendly_scores(y_train[len(y_train)-sample_size:], target_prompt)

    Y_train = np.concatenate((source_y, target_y))
    scaled_train_mean = Y_train.mean(axis=0)
    if dev_size == -1:
        Y_dev = utils.get_model_friendly_scores(y_dev, source_prompt)
        scaled_dev_mean = utils.get_model_friendly_scores(dev_mean, source_prompt)
    elif dev_size >= 0:
        Y_dev = utils.get_model_friendly_scores(y_dev, target_prompt)
        scaled_dev_mean = utils.get_model_friendly_scores(dev_mean, target_prompt)
    else:
        raise NotImplementedError
    Y_test = utils.get_model_friendly_scores(y_test, target_prompt)
    scaled_test_mean = utils.get_model_friendly_scores(test_mean, target_prompt)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    if need_context:
        logger.info('  train context shape: ' + str(train_context.shape))
        logger.info('  dev context shape: ' + str(dev_context.shape))
        logger.info('  test context shape: ' + str(test_context.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean after scaling: %s' %
                (str(scaled_train_mean)))
    logger.info('  dev_y mean: %s, stdev: %s, dev_y mean after scaling: %s' %
                (str(dev_mean), str(dev_std), str(scaled_dev_mean)))
    logger.info('  test_y mean: %s, stdev: %s, test_y mean after scaling: %s' %
                (str(test_mean), str(test_std), str(scaled_test_mean)))

    if embedding_path:
        emb_dict, emb_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, emb_dim)
        emb_matrix = utils.build_embedding_table(vocab, emb_dict, emb_dim, logger, caseless=True)
    else:
        emb_matrix = None

    return (X_train, Y_train, mask_train, train_context, train_text, train_ids), \
           (X_dev, Y_dev, mask_dev, dev_context, dev_text, dev_ids), \
           (X_test, Y_test, mask_test, test_context, test_text, test_ids), \
           vocab, len(vocab), emb_matrix, overall_maxlen, overall_maxnum, scaled_train_mean, context_len, context_num


def prepare_sentence_data(
        datapaths,
        embedding_path=None,
        embedding='word2vec',
        emb_dim=100,
        prompt_id=1,
        vocab_size=0,
        tokenize_text=True,
        to_lower=True,
        replace_num=True,
        vocab_path=None,
        score_index=6,
        need_context=True,
        sentence_level=True
):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, train_y, train_ids, train_text), \
    (dev_x, dev_y, dev_ids, dev_text), \
    (test_x, test_y, test_ids, test_text), \
    vocab, overall_maxlen, overall_maxnum = \
        reader.get_data(
            datapaths,
            prompt_id,
            vocab_size,
            tokenize_text,
            to_lower,
            replace_num,
            vocab_path,
            score_index,
            sentence_level
        )

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overall_maxnum, overall_maxlen,
                                                                    post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overall_maxnum, overall_maxlen,
                                                              post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overall_maxnum, overall_maxlen,
                                                                 post_padding=True)

    if need_context:
        context, context_len, context_num, _ = reader.get_context(prompt_id, vocab, to_lower)
    else:
        # Dummy context
        context = [[0]]
        context_len = 1
        context_num = 1
    train_context = [context] * len(train_x)
    dev_context = [context] * len(dev_x)
    test_context = [context] * len(test_x)

    train_context, _, _ = utils.padding_sentence_sequences(train_context, train_y, context_num, context_len, post_padding=True)
    dev_context, _, _ = utils.padding_sentence_sequences(dev_context, dev_y, context_num, context_len, post_padding=True)
    test_context, _, _ = utils.padding_sentence_sequences(test_context, test_y, context_num, context_len, post_padding=True)

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)

    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    Y_train = utils.get_model_friendly_scores(y_train, prompt_id)
    Y_dev = utils.get_model_friendly_scores(y_dev, prompt_id)
    Y_test = utils.get_model_friendly_scores(y_test, prompt_id)
    scaled_train_mean = utils.get_model_friendly_scores(train_mean, prompt_id)
    scaled_dev_mean = utils.get_model_friendly_scores(dev_mean, prompt_id)
    scaled_test_mean = utils.get_model_friendly_scores(test_mean, prompt_id)
    # print Y_train.shape

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    if need_context:
        logger.info('  train context shape: ' + str(train_context.shape))
        logger.info('  dev context shape: ' + str(dev_context.shape))
        logger.info('  test context shape: ' + str(test_context.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))
    logger.info('  dev_y mean: %s, stdev: %s, dev_y mean after scaling: %s' %
                (str(dev_mean), str(dev_std), str(scaled_dev_mean)))
    logger.info('  test_y mean: %s, stdev: %s, test_y mean after scaling: %s' %
                (str(test_mean), str(test_std), str(scaled_test_mean)))

    if embedding_path:
        emb_dict, emb_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, emb_dim)
        emb_matrix = utils.build_embedding_table(vocab, emb_dict, emb_dim, logger, caseless=True)
    else:
        emb_matrix = None

    return (X_train, Y_train, mask_train, train_context, train_text, train_ids), \
           (X_dev, Y_dev, mask_dev, dev_context, dev_text, dev_ids), \
           (X_test, Y_test, mask_test, test_context, test_text, test_ids), \
           vocab, len(vocab), emb_matrix, overall_maxlen, overall_maxnum, scaled_train_mean, context_len, context_num


def get_word_count(text):
    y = []
    for essay in text:
        count = 0
        for sent in essay:
            count = count + len(sent)
        y.append(count)
    return y


def cal_lda_sim(lda_outs, context_out):
    y = []
    for lda_out in lda_outs:
        y.append(np.inner(lda_out, context_out) / (np.linalg.norm(lda_out) * np.linalg.norm(context_out)))
        # y.append(np.dot(lda_out, context_out))

    return y


def get_lda_sim(prompt_id, train_text, dev_text, test_text, context_text, topic_model_mode, lda_len, fold=''):
    filename = 'tc_features/lda_' + str(prompt_id) + '_' + topic_model_mode + '_' + str(lda_len) + fold + '.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            train_y, dev_y, test_y = pickle.load(f)
    else:
        # hardcode to false, need to be a parameter in the future
        normalize = False

        topic_model = TopicModel(
            model_mode=topic_model_mode,
            infer_doc_level=True,
            remove_stopwords=True,
            deacc=True,
            replace_num=True,
            lemmatize=True,
            lda_len=lda_len,
            normalize=normalize
        )

        topic_model.init_model(train_text + [context_text])

        train_out, dev_out, test_out, context_out = topic_model.get_doc_topics([train_text, dev_text, test_text, [context_text]])

        train_y = cal_lda_sim(train_out, context_out[0])
        dev_y = cal_lda_sim(dev_out, context_out[0])
        test_y = cal_lda_sim(test_out, context_out[0])

        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                pickle.dump([train_y, dev_y, test_y], f)

    return [train_y, dev_y, test_y]


def prepare_sentence_data_with_wc(
        datapaths,
        embedding_path=None,
        embedding='word2vec',
        emb_dim=100,
        prompt_id=1,
        vocab_size=0,
        tokenize_text=True,
        to_lower=True,
        replace_num=True,
        vocab_path=None,
        score_index=6,
        need_context=True,
        sentence_level=True,
        signal='wc'
):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, ori_train_y, train_ids, train_text), \
    (dev_x, ori_dev_y, dev_ids, dev_text), \
    (test_x, ori_test_y, test_ids, test_text), \
    vocab, overall_maxlen, overall_maxnum = \
        reader.get_data(
            datapaths,
            prompt_id,
            vocab_size,
            tokenize_text,
            to_lower,
            replace_num,
            vocab_path,
            score_index,
            sentence_level
        )

    if need_context:
        context, context_len, context_num, context_text = reader.get_context(prompt_id, vocab, to_lower)
    else:
        # Dummy context
        context = [[0]]
        context_len = 1
        context_num = 1
    train_context = [context] * len(train_x)
    dev_context = [context] * len(dev_x)
    test_context = [context] * len(test_x)

    if signal == 'wc':
        train_y = get_word_count(train_text)
        dev_y = get_word_count(dev_text)
        test_y = get_word_count(test_text)
    elif signal == 'lda':
        topic_model_mode = 'lda'
        lda_len = 8
        train_y, dev_y, test_y = get_lda_sim(
            prompt_id,
            train_text,
            dev_text,
            test_text,
            context_text,
            topic_model_mode,
            lda_len
        )
    elif signal == 'lda_coh':
        topic_model_mode = 'lda_coh'
        # this lda_len is useless because lad_coh select best num_topics via coherence score
        lda_len = 0
        train_y, dev_y, test_y = get_lda_sim(
            prompt_id,
            train_text,
            dev_text,
            test_text,
            context_text,
            topic_model_mode,
            lda_len
        )
    else:
        raise NotImplementedError

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overall_maxnum, overall_maxlen,
                                                                    post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, dev_y, overall_maxnum, overall_maxlen,
                                                              post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, test_y, overall_maxnum, overall_maxlen,
                                                                 post_padding=True)

    train_context, _, _ = utils.padding_sentence_sequences(train_context, train_y, context_num, context_len, post_padding=True)
    dev_context, _, _ = utils.padding_sentence_sequences(dev_context, dev_y, context_num, context_len, post_padding=True)
    test_context, _, _ = utils.padding_sentence_sequences(test_context, test_y, context_num, context_len, post_padding=True)

    corr = np.corrcoef(y_train.reshape(len(y_train)), ori_train_y)[0][1]
    logger.info('Correlation between LDA sim and evidence scores is %s.' % (corr))

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)

    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    if signal == 'wc':
        # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
        Y_train = utils.get_model_friendly_scores(y_train, prompt_id, use_wc=True)
        Y_dev = utils.get_model_friendly_scores(y_dev, prompt_id, use_wc=True)
        Y_test = utils.get_model_friendly_scores(y_test, prompt_id, use_wc=True)
        scaled_train_mean = utils.get_model_friendly_scores(train_mean, prompt_id, use_wc=True)
        scaled_dev_mean = utils.get_model_friendly_scores(dev_mean, prompt_id, use_wc=True)
        scaled_test_mean = utils.get_model_friendly_scores(test_mean, prompt_id, use_wc=True)
    elif signal == 'lda':
        Y_train = y_train
        Y_dev = y_dev
        Y_test = y_test
        scaled_train_mean = train_mean
        scaled_dev_mean = dev_mean
        scaled_test_mean = test_mean
    elif signal == 'lda_coh':
        Y_train = y_train
        Y_dev = y_dev
        Y_test = y_test
        scaled_train_mean = train_mean
        scaled_dev_mean = dev_mean
        scaled_test_mean = test_mean
    else:
        raise NotImplementedError

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    if need_context:
        logger.info('  train context shape: ' + str(train_context.shape))
        logger.info('  dev context shape: ' + str(dev_context.shape))
        logger.info('  test context shape: ' + str(test_context.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))
    logger.info('  dev_y mean: %s, stdev: %s, dev_y mean after scaling: %s' %
                (str(dev_mean), str(dev_std), str(scaled_dev_mean)))
    logger.info('  test_y mean: %s, stdev: %s, test_y mean after scaling: %s' %
                (str(test_mean), str(test_std), str(scaled_test_mean)))

    if embedding_path:
        emb_dict, emb_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, emb_dim)
        emb_matrix = utils.build_embedding_table(vocab, emb_dict, emb_dim, logger, caseless=True)
    else:
        emb_matrix = None

    return (X_train, Y_train, mask_train, train_context, train_text, train_ids), \
           (X_dev, Y_dev, mask_dev, dev_context, dev_text, dev_ids), \
           (X_test, Y_test, mask_test, test_context, test_text, test_ids), \
           vocab, len(vocab), emb_matrix, overall_maxlen, overall_maxnum, scaled_train_mean, context_len, context_num


def prepare_sentence_data_with_weak_super(
        datapaths,
        embedding_path=None,
        embedding='word2vec',
        emb_dim=100,
        prompt_id=1,
        vocab_size=0,
        tokenize_text=True,
        to_lower=True,
        replace_num=True,
        vocab_path=None,
        score_index=6,
        need_context=True,
        sentence_level=True,
        signal='wc',
        fold='0'
):
    assert len(datapaths) == 3, "data paths should include train, dev and test path"
    (train_x, ori_train_y, train_ids, train_text), \
    (dev_x, ori_dev_y, dev_ids, dev_text), \
    (test_x, ori_test_y, test_ids, test_text), \
    vocab, overall_maxlen, overall_maxnum = \
        reader.get_data(
            datapaths,
            prompt_id,
            vocab_size,
            tokenize_text,
            to_lower,
            replace_num,
            vocab_path,
            score_index,
            sentence_level
        )

    if need_context:
        context, context_len, context_num, context_text = reader.get_context(prompt_id, vocab, to_lower)
    else:
        # Dummy context
        context = [[0]]
        context_len = 1
        context_num = 1
    train_context = [context] * len(train_x)
    dev_context = [context] * len(dev_x)
    test_context = [context] * len(test_x)

    if signal == 'wc':
        train_y = get_word_count(train_text)
    elif signal == 'lda':
        topic_model_mode = 'lda'
        if prompt_id == 1:
            lda_len = 7
        elif prompt_id == 7:
            lda_len = 14
        else:
            raise NotImplementedError
        ver = '_' + fold + '_'
        train_y, _, _ = get_lda_sim(
            prompt_id,
            train_text,
            dev_text,
            test_text,
            context_text,
            topic_model_mode,
            lda_len,
            fold=ver
        )
    elif signal == 'lda_coh':
        topic_model_mode = 'lda_coh'
        # this lda_len is useless because lad_coh select best num_topics via coherence score
        lda_len = 0
        ver = '_' + fold + '_'
        train_y, _, _ = get_lda_sim(
            prompt_id,
            train_text,
            dev_text,
            test_text,
            context_text,
            topic_model_mode,
            lda_len,
            fold=ver
        )
    else:
        raise NotImplementedError

    X_train, y_train, mask_train = utils.padding_sentence_sequences(train_x, train_y, overall_maxnum, overall_maxlen,
                                                                    post_padding=True)
    X_dev, y_dev, mask_dev = utils.padding_sentence_sequences(dev_x, ori_dev_y, overall_maxnum, overall_maxlen,
                                                              post_padding=True)
    X_test, y_test, mask_test = utils.padding_sentence_sequences(test_x, ori_test_y, overall_maxnum, overall_maxlen,
                                                                 post_padding=True)

    train_context, _, _ = utils.padding_sentence_sequences(train_context, train_y, context_num, context_len, post_padding=True)
    dev_context, _, _ = utils.padding_sentence_sequences(dev_context, ori_dev_y, context_num, context_len, post_padding=True)
    test_context, _, _ = utils.padding_sentence_sequences(test_context, ori_test_y, context_num, context_len, post_padding=True)

    corr = np.corrcoef(y_train.reshape(len(y_train)), ori_train_y)[0][1]
    logger.info('Correlation between weak scores and evidence scores is %s.' % (corr))

    train_mean = y_train.mean(axis=0)
    train_std = y_train.std(axis=0)
    dev_mean = y_dev.mean(axis=0)
    dev_std = y_dev.std(axis=0)
    test_mean = y_test.mean(axis=0)
    test_std = y_test.std(axis=0)

    # We need the dev and test sets in the original scale for evaluation
    # dev_y_org = y_dev.astype(reader.get_ref_dtype())
    # test_y_org = y_test.astype(reader.get_ref_dtype())

    if signal == 'wc':
        # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
        low = np.min(y_train)
        high = np.max(y_train)
        Y_train = utils.get_model_friendly_scores(y_train, prompt_id, min_score=low, max_score=high)

        scaled_train_mean = utils.get_model_friendly_scores(train_mean, prompt_id, min_score=low, max_score=high)
    elif signal == 'lda':
        Y_train = y_train
        scaled_train_mean = train_mean
    elif signal == 'lda_coh':
        Y_train = y_train
        scaled_train_mean = train_mean
    else:
        raise NotImplementedError

    Y_dev = utils.get_model_friendly_scores(y_dev, prompt_id, min_score=0, max_score=3)
    Y_test = utils.get_model_friendly_scores(y_test, prompt_id, min_score=0, max_score=3)

    scaled_dev_mean = utils.get_model_friendly_scores(dev_mean, prompt_id, min_score=0, max_score=3)
    scaled_test_mean = utils.get_model_friendly_scores(test_mean, prompt_id, min_score=0, max_score=3)

    logger.info('Statistics:')

    logger.info('  train X shape: ' + str(X_train.shape))
    logger.info('  dev X shape:   ' + str(X_dev.shape))
    logger.info('  test X shape:  ' + str(X_test.shape))

    if need_context:
        logger.info('  train context shape: ' + str(train_context.shape))
        logger.info('  dev context shape: ' + str(dev_context.shape))
        logger.info('  test context shape: ' + str(test_context.shape))

    logger.info('  train Y shape: ' + str(Y_train.shape))
    logger.info('  dev Y shape:   ' + str(Y_dev.shape))
    logger.info('  test Y shape:  ' + str(Y_test.shape))

    logger.info('  train_y mean: %s, stdev: %s, train_y mean after scaling: %s' %
                (str(train_mean), str(train_std), str(scaled_train_mean)))
    logger.info('  dev_y mean: %s, stdev: %s, dev_y mean after scaling: %s' %
                (str(dev_mean), str(dev_std), str(scaled_dev_mean)))
    logger.info('  test_y mean: %s, stdev: %s, test_y mean after scaling: %s' %
                (str(test_mean), str(test_std), str(scaled_test_mean)))

    if embedding_path:
        emb_dict, emb_dim, _ = utils.load_word_embedding_dict(embedding, embedding_path, vocab, logger, emb_dim)
        emb_matrix = utils.build_embedding_table(vocab, emb_dict, emb_dim, logger, caseless=True)
    else:
        emb_matrix = None

    return (X_train, Y_train, mask_train, train_context, train_text, train_ids), \
           (X_dev, Y_dev, mask_dev, dev_context, dev_text, dev_ids), \
           (X_test, Y_test, mask_test, test_context, test_text, test_ids), \
           vocab, len(vocab), emb_matrix, overall_maxlen, overall_maxnum, scaled_train_mean, context_len, context_num

