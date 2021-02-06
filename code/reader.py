# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/11/19 12:58 PM

import random
import codecs
import sys
import nltk
# import logging
import re
import numpy as np
import pickle as pk
import utils
import nltk.data
import json
from gensim.corpora import Dictionary


url_replacer = '<url>'
logger = utils.get_logger("Loading data...")
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

MAX_SENTLEN = 50
MAX_SENTNUM = 100


def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def is_number(token):
    return bool(num_regex.match(token))


def load_vocab(vocab_path):
    logger.info('Loading vocabulary from: ' + vocab_path)
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pk.load(vocab_file)
    return vocab


def read_corpus(file_path, prompt_id, tokenize_text, to_lower, replace_num, word_freqs, sample_size=-1, shuffled_id=''):
    if sample_size == 0:
        return [], 0

    if sample_size > 0:
        with open(shuffled_id) as f:
            json_obj = json.load(f)
            sample_ids = json_obj[str(prompt_id)]
            # if sample_size is larger than the corpus size, use corpus size
            sample_size = min(sample_size, len(sample_ids))
            sample_ids = sample_ids[0: sample_size]

    logger.info('Creating vocabulary from: ' + file_path)
    corpus = []
    total_words = 0
    with codecs.open(file_path, mode='r', encoding='utf-8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[6])
            if essay_set == prompt_id or prompt_id <= 0:
                if sample_size > 0:
                    if essay_id not in sample_ids:
                        continue

                if tokenize_text:
                    content = text_tokenizer(content, True, True, True)
                if to_lower:
                    content = [w.lower() for w in content]
                if replace_num:
                    content = [w for w in content if not is_number(w)]
                corpus.append(content)
                for word in content:
                    try:
                        word_freqs[word] += 1
                    except KeyError:
                        word_freqs[word] = 1
                    total_words += 1
    return corpus, total_words


def create_vocab_from_corpus(corpus, replace_num, vocab_size):
    vocab = Dictionary(corpus)
    if replace_num:
        special_tokens = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    else:
        special_tokens = {'<pad>': 0, '<unk>': 1}
    if vocab_size <= 0:
        # Choose vocab size automatically by removing all singletons
        vocab.filter_extremes(no_below=2, no_above=1.1, keep_n=len(vocab))
    else:
        vocab.filter_extremes(no_below=2, no_above=1.1, keep_n=vocab_size - 3)

    # must perform patch after filter, because this messes up the original index.
    vocab.patch_with_special_tokens(special_tokens)
    assert vocab[0] == '<pad>'
    assert vocab[1] == '<unk>'
    assert vocab[2] == '<num>'

    # import operator
    # sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    # if vocab_size <= 0:
    #     # Choose vocab size automatically by removing all singletons
    #     vocab_size = 0
    #     for word, freq in sorted_word_freqs:
    #         if freq > 1:
    #             vocab_size += 1
    # vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    # vcb_len = len(vocab)
    # index = vcb_len
    # for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
    #     vocab[word] = index
    #     index += 1

    return vocab

def create_cross_vocab(paths, source_prompt, target_prompt, sample_size, dev_size, vocab_size, tokenize_text, to_lower, replace_num):
    train_path, dev_path, test_path, shuffled_id = paths[0], paths[1], paths[2], paths[3]
    word_freqs = {}
    if dev_size == -1:
        source_train_corpus, source_train_total_words = read_corpus(train_path, source_prompt, tokenize_text, to_lower, replace_num, word_freqs)
        source_test_corpus, source_test_total_words = read_corpus(test_path, source_prompt, tokenize_text, to_lower, replace_num, word_freqs)
        corpus = source_train_corpus + source_test_corpus
        total_words = source_train_total_words + source_test_total_words
    elif dev_size >= 0:
        source_train_corpus, source_train_total_words = read_corpus(train_path, source_prompt, tokenize_text, to_lower, replace_num, word_freqs)
        source_dev_corpus, source_dev_total_words = read_corpus(dev_path, source_prompt, tokenize_text, to_lower, replace_num, word_freqs)
        source_test_corpus, source_test_total_words = read_corpus(test_path, source_prompt, tokenize_text, to_lower, replace_num, word_freqs)
        target_train_corpus, target_train_total_words = read_corpus(train_path, target_prompt, tokenize_text, to_lower, replace_num, word_freqs, sample_size, shuffled_id)

        corpus = source_train_corpus + source_dev_corpus + source_test_corpus + target_train_corpus
        total_words = source_train_total_words + source_dev_total_words + source_test_total_words + target_train_total_words

    logger.info('  %i total words, %i unique words' % (total_words, len(word_freqs)))

    return create_vocab_from_corpus(corpus, replace_num, vocab_size)


def create_vocab(file_path, prompt_id, vocab_size, tokenize_text, to_lower, replace_num):
    word_freqs = {}
    corpus, total_words = read_corpus(file_path, prompt_id, tokenize_text, to_lower, replace_num, word_freqs)
    logger.info('  %i total words, %i unique words' % (total_words, len(word_freqs)))

    return create_vocab_from_corpus(corpus, replace_num, vocab_size)


def read_transformer_dataset(file_path, prompt_id, sent_detector, tokenizer, score_index=6):
    logger.info('Reading dataset from: ' + file_path)
    data_x, data_y, text = [], [], []
    max_sentnum = -1
    max_sentlen = -1
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                # tokenize text into sentences
                sent_tokens = transformer_tokenize(content, sent_detector, tokenizer)

                sents_ids = []
                sents_plain =[]
                for sent in sent_tokens:
                    length = len(sent)
                    if length > 0:
                        # because bert and albert model need cls and sep.
                        if max_sentlen < length + 2:
                            max_sentlen = length + 2

                        sent_plain = ["[CLS]"] + sent + ["[SEP]"]
                        sent_ids = tokenizer.convert_tokens_to_ids(sent_plain)

                        sents_ids.append(sent_ids)
                        sents_plain.append(sent_plain)

                data_x.append(sents_ids)
                data_y.append(score)
                text.append(sents_plain)

                if max_sentnum < len(sents_ids):
                    max_sentnum = len(sents_ids)
    return data_x, data_y, text, max_sentlen, max_sentnum


def read_dataset(file_path, prompt_id, vocab, to_lower, replace_num, score_index=6, sentence_level=True, sample_size=-1, shuffled_id=''):
    if sample_size == 0:
        return [], [], [], 0, 0, []

    if sample_size > 0:
        with open(shuffled_id) as f:
            json_obj = json.load(f)
            sample_ids = json_obj[str(prompt_id)]
            # if sample_size is larger than the corpus size, use corpus size
            sample_size = min(sample_size, len(sample_ids))
            sample_ids = sample_ids[0: sample_size]

    logger.info('Reading dataset from: ' + file_path)
    data_x, data_y, essay_ids, text = [], [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    max_sentnum = -1
    max_sentlen = -1
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[score_index])
            if essay_set == prompt_id or prompt_id <= 0:
                if sample_size > 0:
                    if essay_id not in sample_ids:
                        continue
                # tokenize text into sentences
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=sentence_level)
                if to_lower:
                    sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

                sent_indices = []
                indices = []
                sentences = []
                words = []
                for sent in sent_tokens:
                    length = len(sent)
                    if length > 0:
                        if max_sentlen < length:
                            max_sentlen = length
                        for word in sent:
                            if replace_num and is_number(word):
                                indices.append('<num>')
                                num_hit += 1
                            else:
                                indices.append(word)
                            # elif word in word_list:
                            #     indices.append(word)
                            # else:
                            #     indices.append('<unk>')
                            #     unk_hit += 1
                            total += 1
                            words.append(word)
                        sentences.append(words)
                        indices = vocab.doc2idx(indices, 1)
                        unk_hit += indices.count(1)
                        sent_indices.append(indices)
                        indices = []
                        words = []
                text.append(sentences)
                data_x.append(sent_indices)
                data_y.append(score)
                essay_ids.append(essay_id)

                if max_sentnum < len(sent_indices):
                    max_sentnum = len(sent_indices)
    if replace_num:
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    else:
        logger.info('  <unk> hit rate: %.2f%%' % (100 * unk_hit / total))
    return data_x, data_y, essay_ids, max_sentlen, max_sentnum, text


def get_transformer_data(
        paths,
        prompt_id,
        transformer_path,
        score_index=6
):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]
    logger.info("Prompt id is %s" % prompt_id)

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = utils.create_tokenizer_from_transformers(transformer_path)

    train_x, train_y, train_text, train_maxsentlen, train_maxsentnum = read_transformer_dataset(train_path, prompt_id,
                                                                                                sent_detector, tokenizer,
                                                                                                score_index)
    dev_x, dev_y, dev_text, dev_maxsentlen, dev_maxsentnum= read_transformer_dataset(dev_path, prompt_id, sent_detector,
                                                                                     tokenizer, score_index)
    test_x, test_y, test_text, test_maxsentlen, test_maxsentnum = read_transformer_dataset(test_path, prompt_id,
                                                                                           sent_detector, tokenizer,
                                                                                           score_index)

    overall_maxlen = max(train_maxsentlen, dev_maxsentlen, test_maxsentlen)
    overall_maxnum = max(train_maxsentnum, dev_maxsentnum, test_maxsentnum)

    logger.info("Training data max sentence num = %s, max sentence length = %s" % (train_maxsentnum, train_maxsentlen))
    logger.info("Dev data max sentence num = %s, max sentence length = %s" % (dev_maxsentnum, dev_maxsentlen))
    logger.info("Test data max sentence num = %s, max sentence length = %s" % (test_maxsentnum, test_maxsentlen))
    logger.info("Overall max sentence num = %s, max sentence length = %s" % (overall_maxnum, overall_maxlen))

    return (train_x, train_y, train_text), \
           (dev_x, dev_y, dev_text), \
           (test_x, test_y, test_text), \
           overall_maxlen, overall_maxnum



def get_cross_data(
        paths,
        source_prompt,
        target_prompt,
        sample_size,
        dev_size,
        vocab_size,
        tokenize_text=True,
        to_lower=True,
        replace_num=True,
        vocab_path=None,
        score_index=6,
        sentence_level=True
):
    train_path, dev_path, test_path, shuffled_id = paths[0], paths[1], paths[2], paths[3]
    logger.info("Prompt id is %s->%s" % (source_prompt, target_prompt))
    if not vocab_path:
        vocab = create_cross_vocab(paths, source_prompt, target_prompt, sample_size, dev_size, vocab_size, tokenize_text, to_lower, replace_num)
        if len(vocab) < vocab_size:
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))

    if dev_size == -1:
        train_x_s, train_y_s, train_ids_s, train_maxsentlen_s, train_maxsentnum_s, train_text_s = \
            read_dataset(train_path, source_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        dev_x, dev_y, dev_ids, dev_maxsentlen, dev_maxsentnum, dev_text = \
            read_dataset(dev_path, source_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        test_x_s, test_y_s, test_ids_s, test_maxsentlen_s, test_maxsentnum_s, test_text_s = \
            read_dataset(test_path, source_prompt, vocab, to_lower, replace_num, score_index, sentence_level)

        train_x = train_x_s + test_x_s
        train_y = train_y_s + test_y_s
        train_ids = train_ids_s + test_ids_s
        train_text = train_text_s + test_text_s

        train_maxsentlen = max(train_maxsentlen_s, test_maxsentlen_s)
        train_maxsentnum = max(train_maxsentnum_s, test_maxsentnum_s)

        train_x_t, train_y_t, train_ids_t, train_maxsentlen_t, train_maxsentnum_t, train_text_t = \
            read_dataset(train_path, target_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        dev_x_t, dev_y_t, dev_ids_t, dev_maxsentlen_t, dev_maxsentnum_t, dev_text_t = \
            read_dataset(dev_path, target_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        test_x_t, test_y_t, test_ids_t, test_maxsentlen_t, test_maxsentnum_t, test_text_t = \
            read_dataset(test_path, target_prompt, vocab, to_lower, replace_num, score_index, sentence_level)

        test_x = train_x_t + dev_x_t + test_x_t
        test_y = train_y_t + dev_y_t + test_y_t
        test_ids = train_ids_t + dev_ids_t + test_ids_t
        test_text = train_text_t + dev_text_t + test_text_t

        test_maxsentlen = max(train_maxsentlen_t, dev_maxsentlen_t, test_maxsentlen_t)
        test_maxsentnum = max(train_maxsentnum_t, dev_maxsentnum_t, test_maxsentnum_t)

    elif dev_size >= 0:
        train_x_s, train_y_s, train_ids_s, train_maxsentlen_s, train_maxsentnum_s, train_text_s = \
            read_dataset(train_path, source_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        dev_x_s, dev_y_s, dev_ids_s, dev_maxsentlen_s, dev_maxsentnum_s, dev_text_s = \
            read_dataset(dev_path, source_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        test_x_s, test_y_s, test_ids_s, test_maxsentlen_s, test_maxsentnum_s, test_text_s = \
            read_dataset(test_path, source_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        train_x_t, train_y_t, train_ids_t, train_maxsentlen_t, train_maxsentnum_t, train_text_t = \
            read_dataset(train_path, target_prompt, vocab, to_lower, replace_num, score_index, sentence_level, sample_size, shuffled_id)

        train_x = train_x_s + dev_x_s + test_x_s + train_x_t
        train_y = train_y_s + dev_y_s + test_y_s + train_y_t
        train_ids = train_ids_s + dev_ids_s + test_ids_s + train_ids_t
        train_text = train_text_s + dev_text_s + test_text_s + train_text_t
        train_maxsentlen = max(train_maxsentlen_s, dev_maxsentlen_s, test_maxsentlen_s, train_maxsentlen_t)
        train_maxsentnum = max(train_maxsentnum_s, dev_maxsentnum_s, test_maxsentnum_s, train_maxsentnum_t)

        dev_x, dev_y, dev_ids, dev_maxsentlen, dev_maxsentnum, dev_text = \
            read_dataset(dev_path, target_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
        if dev_size > 0 and dev_size < len(dev_x):
            indices = np.random.choice(len(dev_x), dev_size, replace=False)
            dev_x = np.array(dev_x, dtype=object)[indices].tolist()
            dev_y = np.array(dev_y, dtype=object)[indices].tolist()
            dev_ids = np.array(dev_ids, dtype=object)[indices].tolist()
            dev_text = np.array(dev_text, dtype=object)[indices].tolist()

            # update dev_maxsentlen and dev_maxsentnum
            dev_maxsentlen = 0
            dev_maxsentnum = 0
            for essay in dev_x:
                if len(essay) > dev_maxsentnum:
                    dev_maxsentnum = len(essay)
                for sent in essay:
                    if len(sent) > dev_maxsentlen:
                        dev_maxsentlen = len(sent)

        test_x, test_y, test_ids, test_maxsentlen, test_maxsentnum, test_text = \
            read_dataset(test_path, target_prompt, vocab, to_lower, replace_num, score_index, sentence_level)
    else:
        raise NotImplementedError

    overall_maxlen = max(train_maxsentlen, dev_maxsentlen, test_maxsentlen)
    overall_maxnum = max(train_maxsentnum, dev_maxsentnum, test_maxsentnum)

    logger.info("Training data max sentence num = %s, max sentence length = %s" % (train_maxsentnum, train_maxsentlen))
    logger.info("Dev data max sentence num = %s, max sentence length = %s" % (dev_maxsentnum, dev_maxsentlen))
    logger.info("Test data max sentence num = %s, max sentence length = %s" % (test_maxsentnum, test_maxsentlen))
    logger.info("Overall max sentence num = %s, max sentence length = %s" % (overall_maxnum, overall_maxlen))

    return (train_x, train_y, train_ids, train_text), \
           (dev_x, dev_y, dev_ids, dev_text), \
           (test_x, test_y, test_ids, test_text), \
           vocab, overall_maxlen, overall_maxnum

def get_data(
        paths,
        prompt_id,
        vocab_size,
        tokenize_text=True,
        to_lower=True,
        replace_num=True,
        vocab_path=None,
        score_index=6,
        sentence_level=True
):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    logger.info("Prompt id is %s" % prompt_id)
    if not vocab_path:
        vocab = create_vocab(train_path, prompt_id, vocab_size, tokenize_text, to_lower, replace_num)
        if len(vocab) < vocab_size:
            logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
        else:
            assert vocab_size == 0 or len(vocab) == vocab_size
    else:
        vocab = load_vocab(vocab_path)
        if len(vocab) != vocab_size:
            logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
    logger.info('  Vocab size: %i' % (len(vocab)))

    train_x, train_y, train_ids, train_maxsentlen, train_maxsentnum, train_text = \
        read_dataset(train_path, prompt_id, vocab, to_lower, replace_num, score_index, sentence_level)
    dev_x, dev_y, dev_ids, dev_maxsentlen, dev_maxsentnum, dev_text = \
        read_dataset(dev_path, prompt_id, vocab, to_lower, replace_num, score_index, sentence_level)
    test_x, test_y, test_ids, test_maxsentlen, test_maxsentnum, test_text = \
        read_dataset(test_path, prompt_id, vocab,  to_lower, replace_num, score_index, sentence_level)

    overall_maxlen = max(train_maxsentlen, dev_maxsentlen, test_maxsentlen)
    overall_maxnum = max(train_maxsentnum, dev_maxsentnum, test_maxsentnum)

    logger.info("Training data max sentence num = %s, max sentence length = %s" % (train_maxsentnum, train_maxsentlen))
    logger.info("Dev data max sentence num = %s, max sentence length = %s" % (dev_maxsentnum, dev_maxsentlen))
    logger.info("Test data max sentence num = %s, max sentence length = %s" % (test_maxsentnum, test_maxsentlen))
    logger.info("Overall max sentence num = %s, max sentence length = %s" % (overall_maxnum, overall_maxlen))

    return (train_x, train_y, train_ids, train_text),\
           (dev_x, dev_y, dev_ids, dev_text),\
           (test_x, test_y, test_ids, test_text),\
           vocab, overall_maxlen, overall_maxnum


def get_context(prompt_id, vocab, to_lower=True):
    context_path = 'data/' + str(prompt_id) + '.txt'
    logger.info('Reading context from: ' + context_path)

    num_hit, unk_hit, total = 0., 0., 0.
    max_context_sentlen = -1
    context_sentnum = 0
    context_indices = []
    context_text = []
    with codecs.open(context_path, mode='r', encoding='UTF8') as input_file:
        for line in input_file:
            # context_sent = context_sent + ' ' + line

            # tokenize text into sentences
            context_sent = line
            sent_tokens = text_tokenizer(context_sent, replace_url_flag=True, tokenize_sent_flag=True)
            if to_lower:
                sent_tokens = [[w.lower() for w in s] for s in sent_tokens]


            indices = []
            for sent in sent_tokens:
                length = len(sent)
                if length > 0:
                    if max_context_sentlen < length:
                        max_context_sentlen = length
                    for word in sent:
                        if is_number(word):
                            indices.append('<num>')
                            num_hit += 1
                        else:
                            indices.append(word)
                        # elif word in word_list:
                        #     indices.append(word)
                        # else:
                        #     indices.append('<unk>')
                        #     unk_hit += 1
                        total += 1
                    context_text.append(sent)
                    indices = vocab.doc2idx(indices, 1)
                    unk_hit += indices.count(1)
                    context_indices.append(indices)
                    indices = []






            # indices = []
            # for sent in sent_tokens:
            #     length = len(sent)
            #     if (length > 0) :
            #         if max_context_sentlen < length:
            #             max_context_sentlen = length
            #
            #         for word in sent:
            #             if is_number(word):
            #                 indices.append(vocab['<num>'])
            #                 num_hit += 1
            #             elif word in vocab:
            #                 indices.append(vocab[word])
            #             else:
            #                 indices.append(vocab['<unk>'])
            #                 unk_hit += 1
            #             total += 1
            #         context_indices.append(indices)
            #         indices = []
            # # max_context_sentlen = len(indices)

        # context_indices.append(indices)
        context_sentnum = len(context_indices)

    logger.info("Context sentence num = %s, max sentence length = %s" % (context_sentnum, max_context_sentlen))
    return context_indices, max_context_sentlen, context_sentnum, context_text

def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    if replace_url_flag:
        text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
        # print text
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
        # print text
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
        # print text


    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        # print sent_tokens
        # sys.exit(0)
        # if not create_vocab_flag:
        #     print "After processed and tokenized, sentence num = %s " % len(sent_tokens)
        return sent_tokens
    else:
        return [tokens]


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):

    # tokenize a long text to a list of sentences
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)

    # Note
    # add special preprocessing for abnormal sentence splitting
    # for example, sentence1 entangled with sentence2 because of period "." connect the end of sentence1 and the begin of sentence2
    # see example: "He is running.He likes the sky". This will be treated as one sentence, needs to be specially processed.
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        # print tokens
        return tokens

    # TODO here
    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    # if len(sent_tokens) > 90:
    #     print len(sent_tokens), sent_tokens
    return sent_tokens


def shorten_sentence(sent, max_sentlen):
    # handling extra long sentence, truncate to no more extra max_sentlen
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        # print len(tokens)
        # Step 1: split sentence based on keywords
        # split_keywords = ['because', 'but', 'so', 'then', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = (int) (len(tokens) / max_sentlen)
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        # Step 2: split sentence to no more than max_sentlen
        # if there are still sentences whose length exceeds max_sentlen
        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = (int) (len(token) / max_sentlen)
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
            return [tokens]

    # print "Before processed sentences length = %d, after processed sentences num = %d " % (len(tokens), len(new_tokens))
    return new_tokens


def transformer_tokenize(text, sent_detector, tokenizer, max_seq_length=MAX_SENTLEN):
    sents = sent_detector.tokenize(text.strip())
    sents = [tokenizer.tokenize(sent) for sent in sents]
    sent_tokens=[]
    for sent in sents:
        shorten_sents_tokens = shorten_transformer_sentence(sent, max_seq_length-2)
        sent_tokens.extend(shorten_sents_tokens)
    # if len(sent_tokens) > 90:
    #     print len(sent_tokens), sent_tokens
    return sent_tokens


def shorten_transformer_sentence(tokens, max_sentlen):
    # handling extra long sentence, truncate to no more extra max_sentlen
    new_tokens = []
    if len(tokens) > max_sentlen:
        # print len(tokens)
        # Step 1: split sentence based on keywords
        # split_keywords = ['because', 'but', 'so', 'then', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = (int) (len(tokens) / max_sentlen)
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        # Step 2: split sentence to no more than max_sentlen
        # if there are still sentences whose length exceeds max_sentlen
        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = (int) (len(token) / max_sentlen)
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
        return [tokens]

    #print("Before processed sentences length = %d, after processed sentences num = %d " % (len(tokens), len(new_tokens)))
    return new_tokens
