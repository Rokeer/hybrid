# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   3/11/20 4:26 PM

import nltk
import os
import shlex
import subprocess
import pickle
import numpy as np
import utils
import pandas as pd
from tagsets import discourse_tagset, modal_tagset, pos_tag_tagset, wn_pos_tag_tagset, argumentation_tagset, sent_function_label_tagset
from nltk.parse.corenlp import CoreNLPServer
from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import sentiwordnet as swn, wordnet as wn
from nltk import pos_tag, pos_tag_sents
from nltk.stem import WordNetLemmatizer
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize, word_tokenize

from features.lda.topic_model import TopicModel
from features.argumentation.ag_tagger import Targer


def load_source_pkl(pkl_files, pkl_objs):
    for pkl_file, pkl_obj in zip(pkl_files, pkl_objs):
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                new_pkl_obj = pickle.load(f)
        else:
            new_pkl_obj = {}
        old_len = len(pkl_obj)
        pkl_obj.update(new_pkl_obj)
        assert len(pkl_obj) == old_len + len(new_pkl_obj), 'old_obj and new_obj have same key, please double check.'


def rta_doc(train_ids, dev_ids, test_ids, prompt_id):
    # This is a read only feature. Use Java program AES->summer2020->GenerateIntegrationFeature.java
    filename = 'features/rta/rta_doc_' + str(prompt_id) + '.csv'

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        feature_dict = dict()
        for index, row in df.iterrows():
            feature_vec = []
            feature_vec.append(row['npe'])
            feature_vec.append(row['con'])
            feature_vec.append(row['woc'])
            for i in range(8):
                feature_vec.append(row['spc' + str(i)])
            feature_dict[row['fname']] = feature_vec
    else:
        raise FileNotFoundError

    train_out = np.asarray([feature_dict[text_id] for text_id in train_ids], dtype=np.float)
    dev_out = np.asarray([feature_dict[text_id] for text_id in dev_ids], dtype=np.float)
    test_out = np.asarray([feature_dict[text_id] for text_id in test_ids], dtype=np.float)

    train_scaled, dev_scaled, test_scaled = utils.scale_feature(train_out, dev_out, test_out)

    return train_scaled, dev_scaled, test_scaled


def word_count_doc(train_text, dev_text, test_text):
    train_out = [[sum(sents)] for sents in [[len(sent) for sent in text] for text in train_text]]
    dev_out = [[sum(sents)] for sents in [[len(sent) for sent in text] for text in dev_text]]
    test_out = [[sum(sents)] for sents in [[len(sent) for sent in text] for text in test_text]]

    train_ary = np.asarray(train_out, dtype=np.float)
    dev_ary = np.asarray(dev_out, dtype=np.float)
    test_ary = np.asarray(test_out, dtype=np.float)

    train_scaled, dev_scaled, test_scaled = utils.scale_feature(train_ary, dev_ary, test_ary)

    return train_scaled, dev_scaled, test_scaled


def word_count_sent(train_text, dev_text, test_text, max_sentnum):
    train_out = [[[len(sent)] for sent in text] for text in train_text]
    dev_out = [[[len(sent)] for sent in text] for text in dev_text]
    test_out = [[[len(sent)] for sent in text] for text in test_text]

    train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, 1, dtype=np.float).reshape(-1,1)
    dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, 1, dtype=np.float).reshape(-1,1)
    test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, 1, dtype=np.float).reshape(-1,1)

    train_scaled, dev_scaled, test_scaled = utils.scale_feature(train_padded, dev_padded, test_padded)

    train_reshape = train_scaled.reshape(-1, max_sentnum, 1)
    dev_reshape = dev_scaled.reshape(-1, max_sentnum, 1)
    test_reshape = test_scaled.reshape(-1, max_sentnum, 1)

    return train_reshape, dev_reshape, test_reshape


def find_sent_discourse_label(sent, sent_content, prompt_content, transitional_words, first_sent, last_sent, one_tag=True):
    elaboration = 0
    prompt = 0
    transition = 0
    thesis = 0
    main_idea = 0
    support = 0
    conclusion = 0
    rebuttal = 0
    solution = 0
    suggestion = 0

    if 'they' in sent:
        elaboration = elaboration + 1
    if 'them' in sent:
        elaboration = elaboration + 1
    if 'my' in sent:
        elaboration = elaboration + 1
    if 'he' in sent:
        elaboration = elaboration + 1
    if 'she' in sent:
        elaboration = elaboration + 1

    if first_sent:
        prompt = prompt + 1
    if len(sent_content) > 0:
        overlap = sent_content.intersection(prompt_content)
        prompt = prompt + (5 / 2) * len(overlap) / len(sent_content)

    if '?' in sent:
        transition = transition + 1
    ori_sent = ' '.join(sent)
    for word in transitional_words:
        if ori_sent.find(word) != -1:
            transition = transition + 1
            break

    if 'agree' in sent:
        thesis = thesis + 1
    if 'disagree' in sent:
        thesis = thesis + 1
    if 'think' in sent:
        thesis = thesis + 1
    if 'opinion' in sent:
        thesis = thesis + 1
    if first_sent:
        thesis = thesis + 1

    if 'firstly' in sent:
        main_idea = main_idea + 1
    if 'secondly' in sent:
        main_idea = main_idea + 1
    if 'thirdly' in sent:
        main_idea = main_idea + 1
    if 'another' in sent:
        main_idea = main_idea + 1
    if 'aspect' in sent:
        main_idea = main_idea + 1

    if 'example' in sent:
        support = support + 1
    if 'instance' in sent:
        support = support + 1

    if 'conclusion' in sent:
        conclusion = conclusion + 1
    if 'conclude' in sent:
        conclusion = conclusion + 1
    if 'therefore' in sent:
        conclusion = conclusion + 1
    if 'sum' in sent:
        conclusion = conclusion + 1
    if last_sent:
        conclusion = conclusion + 1

    if 'however' in sent:
        rebuttal = rebuttal + 1
    if 'but' in sent:
        rebuttal = rebuttal + 1
    if 'argue' in sent:
        rebuttal = rebuttal + 1

    if 'solve' in sent:
        solution = solution + 1
    if 'solved' in sent:
        solution = solution + 1
    if 'solution' in sent:
        solution = solution + 1

    if 'should' in sent:
        suggestion = suggestion + 1
    if 'let' in sent:
        suggestion = suggestion + 1
    if 'must' in sent:
        suggestion = suggestion + 1
    if 'ought' in sent:
        suggestion = suggestion + 1

    if one_tag:
        result_list = list()
        result_list.append((elaboration, 9, 'Elaboration'))
        result_list.append((prompt, 8, 'Prompt'))
        result_list.append((transition, 7, 'Transition'))
        result_list.append((thesis, 6, 'Thesis'))
        result_list.append((main_idea, 5, 'MainIdea'))
        result_list.append((support, 4, 'Support'))
        result_list.append((conclusion, 3, 'Conclusion'))
        result_list.append((rebuttal, 2, 'Rebuttal'))
        result_list.append((solution, 1, 'Solution'))
        result_list.append((suggestion, 0, 'Suggestion'))

        result_list = sorted(result_list, reverse=True)
        return [sent_function_label_tagset[result_list[0][2]]]
    else:
        result = [
            elaboration,
            prompt,
            transition,
            thesis,
            main_idea,
            support,
            conclusion,
            rebuttal,
            solution,
            suggestion
        ]
        return result


def get_sent_discourse_func_label(texts, ids, prompts, df_content_words, transitional_words, one_tag):
    results = []
    for text, text_id, prompt in zip(texts, ids, prompts):
        if text_id in df_content_words:
            sent_contents = df_content_words[text_id]
        else:
            sent_contents = []

        result = []
        for sent_ind in range(len(text)):
            sent = text[sent_ind]

            if text_id in df_content_words:
                sent_content = sent_contents[sent_ind]
            else:
                tagged_sentence = pos_tag(sent)
                wn_seq = [(word, penn_to_wn(tag)) for word, tag in tagged_sentence]
                sent_content = set([word for word, tag in wn_seq if tag != ''])
                sent_contents.append(sent_content)

            prompt_content = df_content_words['prompt_' + str(prompt)]

            first_sent = False
            if sent_ind == 0:
                first_sent = True

            last_sent = False
            if sent_ind == len(text) - 1:
                last_sent = True

            result_sent = find_sent_discourse_label(sent, sent_content, prompt_content, transitional_words, first_sent, last_sent, one_tag)
            result.append(result_sent)

        results.append(result)
        df_content_words[text_id] = sent_contents

    return results


def get_prompt_content(prompt_id, df_content_words):
    filename = 'features/discourse_func_label/prompt_' + str(prompt_id) + '.txt'

    pos_tags = []
    with open(filename) as f:
        for line in f:
            sents = sent_tokenize(line)
            if len(sents) == 0:
                continue
            for sent in sents:
                pos_tags = pos_tags + pos_tag(word_tokenize(sent))

    wn_seq = [(word, penn_to_wn(tag)) for word, tag in pos_tags]
    prompt_content = set([word.lower() for word, tag in wn_seq if tag != ''])
    df_content_words['prompt_' + str(prompt_id)] = prompt_content


def read_transitional_words():
    filename = 'features/discourse_func_label/transitional.txt'

    result = []
    with open(filename) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            result.append(line.strip())

    return set(result)


# this might have problem when first run cross prompt.
def discourse_func_sent(train_text, dev_text, test_text, train_ids, dev_ids, test_ids, prompt_id, max_sentnum, one_tag, cross_domain=False, source_prompt=-1, sample_size=0):
    filename = 'data/feature_dicts/df_content_word_' + str(prompt_id) + '.pkl'

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            df_content_words = pickle.load(f)
    else:
        df_content_words = {}

    if 'prompt_' + str(prompt_id) not in df_content_words:
        get_prompt_content(prompt_id, df_content_words)

    train_prompt = [prompt_id for i in range(len(train_ids))]
    dev_prompt = [prompt_id for i in range(len(dev_ids))]
    test_prompt = [prompt_id for i in range(len(test_ids))]

    if cross_domain:
        pkl_files = ['data/feature_dicts/df_content_word_' + str(source_prompt) + '.pkl']
        pkl_objs = [df_content_words]
        load_source_pkl(pkl_files, pkl_objs)

        if 'prompt_' + str(source_prompt) not in df_content_words:
            get_prompt_content(source_prompt, df_content_words)

        train_prompt = [source_prompt for i in range(len(train_ids) - sample_size)] + train_prompt[len(train_ids) - sample_size:]




    transitional_words = read_transitional_words()

    train_out = get_sent_discourse_func_label(train_text, train_ids, train_prompt, df_content_words, transitional_words, one_tag)
    dev_out = get_sent_discourse_func_label(dev_text, dev_ids, dev_prompt, df_content_words, transitional_words, one_tag)
    test_out = get_sent_discourse_func_label(test_text, test_ids, test_prompt, df_content_words, transitional_words, one_tag)

    if one_tag:
        train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, 1).squeeze(-1)
        dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, 1).squeeze(-1)
        test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, 1).squeeze(-1)
    else:
        train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, len(sent_function_label_tagset), dtype=np.float).squeeze(-1)
        dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, len(sent_function_label_tagset), dtype=np.float).squeeze(-1)
        test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, len(sent_function_label_tagset), dtype=np.float).squeeze(-1)

    if not os.path.exists(filename):
        if not cross_domain:
            with open(filename, 'wb') as f:
                pickle.dump(df_content_words, f)

    return train_padded, dev_padded, test_padded


def get_ag_seq(texts, ids, tagger, arguments):
    results = []
    for text, text_id in zip(texts, ids):
        essay_plain = []
        for sent in text:
            essay_plain = essay_plain + sent
        if text_id in arguments:
            tokens, labels, probs = arguments[text_id]
        else:
            tokens, labels, probs = tagger.classify(' '.join(essay_plain))
        essay_labels = []
        for sent_labels in labels:
            essay_labels = essay_labels + sent_labels
        if len(essay_plain) != len(essay_labels):
            raise ValueError

        result = []
        b_index = 0
        for sent in text:
            tmp_labels = essay_labels[b_index:b_index+len(sent)]
            result.append([argumentation_tagset[label] for label in tmp_labels])
            b_index = b_index + len(sent)

        results.append(result)
        arguments[text_id] = [tokens, labels, probs]
        # print(len(results))

    return results


def argumentation_word(train_text, dev_text, test_text, train_ids, dev_ids, test_ids, prompt_id, max_sentnum, max_sentlen, cross_domain=False, source_prompt=-1):
    filename = 'data/feature_dicts/arguments_' + str(prompt_id) + '.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            arguments = pickle.load(f)
    else:
        arguments = {}

    if cross_domain:
        pkl_files = ['data/feature_dicts/arguments_' + str(source_prompt) + '.pkl']
        pkl_objs = [arguments]
        load_source_pkl(pkl_files, pkl_objs)

    tagger = Targer()

    train_out = get_ag_seq(train_text, train_ids, tagger, arguments)
    dev_out = get_ag_seq(dev_text, dev_ids, tagger, arguments)
    test_out = get_ag_seq(test_text, test_ids, tagger, arguments)

    train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, max_sentlen)
    dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, max_sentlen)
    test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, max_sentlen)

    if not os.path.exists(filename):
        if not cross_domain:
            with open(filename, 'wb') as f:
                pickle.dump(arguments, f)

    return train_padded, dev_padded, test_padded


def syllables(word):
    #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count


def nsyl(word, cmudict):
    try:
        # simply return the first result
        return [len(list(y for y in x if y[-1].isdigit())) for x in cmudict[word.lower()]][0]
    except KeyError:
        # if word not found in cmudict
        return syllables(word)


def cal_readability_doc(texts, cmudict):
    results = []
    for text in texts:
        num_sent = len(text)
        num_word = 0
        num_syllable = 0
        for sent in text:
            num_word = num_word + len(sent)
            for word in sent:
                num_syllable = num_syllable + nsyl(word, cmudict)
        fre = 206.835 - 1.015 * (num_word / num_sent) - 84.6 * (num_syllable / num_word)
        results.append(fre)
    return results


def readability_doc(train_text, dev_text, test_text):
    d = cmudict.dict()

    train_out = np.asarray(cal_readability_doc(train_text, d), dtype=np.float).reshape(-1, 1)
    dev_out = np.asarray(cal_readability_doc(dev_text, d), dtype=np.float).reshape(-1, 1)
    test_out = np.asarray(cal_readability_doc(test_text, d), dtype=np.float).reshape(-1, 1)

    train_scaled, dev_scaled, test_scaled = utils.scale_feature(train_out, dev_out, test_out)

    return train_scaled, dev_scaled, test_scaled


def cal_readability_sent(texts, cmudict):
    results = []

    for text in texts:
        result = []
        for sent in text:
            if len(sent) == 0:
                # use the largest value if len(sent) is 0
                result.append(206.836)
                continue
            num_syllable = 0
            for word in sent:
                num_syllable = num_syllable + nsyl(word, cmudict)
            fre = 206.835 - 1.015 * len(sent) - 84.6 * (num_syllable / len(sent))
            result.append([fre])
        results.append(result)

    return results


def readability_sent(train_text, dev_text, test_text, max_sentnum):
    d = cmudict.dict()

    train_out = cal_readability_sent(train_text, d)
    dev_out = cal_readability_sent(dev_text, d)
    test_out = cal_readability_sent(test_text, d)

    train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, 1, dtype=np.float).squeeze(-1)
    dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, 1, dtype=np.float).squeeze(-1)
    test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, 1, dtype=np.float).squeeze(-1)
    return train_padded, dev_padded, test_padded


def lda_doc(train_text, dev_text, test_text, prompt_id, fold, topic_model_mode, lda_len, cross_domain=False, source_prompt=-1, sample_size=0, dev_size=0):
    filename = 'data/feature_dicts/lda_doc_' + str(prompt_id) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
    if cross_domain:
        if dev_size == 0:
            filename = 'data/feature_dicts/lda_doc_' + str(source_prompt) + '_' + str(prompt_id) + '_' + str(sample_size) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
        else:
            filename = 'data/feature_dicts/lda_doc_' + str(source_prompt) + '_' + str(prompt_id) + '_' + str(sample_size) + '_' + str(dev_size) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            train_padded, dev_padded, test_padded = pickle.load(f)
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

        topic_model.init_model(train_text)

        train_out, dev_out, test_out = topic_model.get_doc_topics([train_text, dev_text, test_text])
        train_padded = np.asarray(train_out, dtype=np.float)
        dev_padded = np.asarray(dev_out, dtype=np.float)
        test_padded = np.asarray(test_out, dtype=np.float)

        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                pickle.dump([train_padded, dev_padded, test_padded], f)

    return train_padded, dev_padded, test_padded


def lda_sent(train_text, dev_text, test_text, prompt_id, fold, topic_model_mode, lda_len, cross_domain=False, source_prompt=-1, sample_size=0, dev_size=0):
    filename = 'data/feature_dicts/lda_sent_' + str(prompt_id) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
    if cross_domain:
        if dev_size == 0:
            filename = 'data/feature_dicts/lda_sent_' + str(source_prompt) + '_' + str(prompt_id) + '_' + str(sample_size) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
        else:
            filename = 'data/feature_dicts/lda_sent_' + str(source_prompt) + '_' + str(prompt_id) + '_' + str(sample_size) + '_' + str(dev_size) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            train_padded, dev_padded, test_padded = pickle.load(f)
    else:
        # hardcode to false, need to be a parameter in the future
        normalize = False

        topic_model = TopicModel(
            model_mode=topic_model_mode,
            infer_doc_level=False,
            remove_stopwords=True,
            deacc=True,
            replace_num=True,
            lemmatize=True,
            lda_len=lda_len,
            normalize=normalize
        )

        topic_model.init_model(train_text)

        train_out, dev_out, test_out, max_sentnum = topic_model.get_sent_topics([train_text, dev_text, test_text])
        train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, lda_len, dtype=np.float).squeeze()
        dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, lda_len, dtype=np.float).squeeze()
        test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, lda_len, dtype=np.float).squeeze()

        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                pickle.dump([train_padded, dev_padded, test_padded], f)

    return train_padded, dev_padded, test_padded


def lda_word(train_text, dev_text, test_text, prompt_id, fold, topic_model_mode, lda_len, cross_domain=False, source_prompt=-1, sample_size=0, dev_size=0):
    filename = 'data/feature_dicts/lda_word_' + str(prompt_id) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
    if cross_domain:
        if dev_size == 0:
            filename = 'data/feature_dicts/lda_word_' + str(source_prompt) + '_' + str(prompt_id) + '_' + str(sample_size) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
        else:
            filename = 'data/feature_dicts/lda_word_' + str(source_prompt) + '_' + str(prompt_id) + '_' + str(sample_size) + '_' + str(dev_size) + '_' + fold + '_' + topic_model_mode + '_' + str(lda_len) + '.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            train_padded, dev_padded, test_padded, emb_table, max_sentnum, max_sentlen = pickle.load(f)
    else:
        # hardcode to false, need to be a parameter in the future
        normalize = False

        topic_model = TopicModel(
            model_mode=topic_model_mode,
            infer_doc_level=False,
            remove_stopwords=True,
            deacc=True,
            replace_num=True,
            lemmatize=True,
            lda_len=lda_len,
            normalize=normalize
        )

        topic_model.init_model(train_text)
        train_out, dev_out, test_out, max_sentnum, max_sentlen = topic_model.get_preprocessed_essays([train_text, dev_text, test_text])
        train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, max_sentlen)
        dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, max_sentlen)
        test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, max_sentlen)
        emb_table = topic_model.get_topic_emb_table()

        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                pickle.dump([train_padded, dev_padded, test_padded, emb_table, max_sentnum, max_sentlen], f)

    return train_padded, dev_padded, test_padded, emb_table, max_sentnum, max_sentlen


def extract_pos_tag_seq(texts, ids, pos_tag_seqs, penn_tree_tags):
    outputs = []

    for text, text_id in zip(texts, ids):
        if text_id in pos_tag_seqs:
            output = pos_tag_seqs[text_id]
        else:
            output = []
            for sent in text:
                tagged_sentence = pos_tag(sent)
                if penn_tree_tags:
                    new_sent = [pos_tag_tagset[tag] for word, tag in tagged_sentence]
                else:
                    new_sent = [wn_pos_tag_tagset[penn_to_wn(tag)] for word, tag in tagged_sentence]
                output.append(new_sent)
            pos_tag_seqs[text_id] = output
        outputs.append(output)
    return outputs


def pos_tag_seq_word(train_text, dev_text, test_text, train_ids, dev_ids, test_ids, prompt_id, max_sentnum, max_sentlen, penn_tree_tags=True, cross_domain=False, source_prompt=-1):
    if penn_tree_tags:
        filename = 'data/feature_dicts/pos_tag_seqs_' + str(prompt_id) + '.pkl'
    else:
        filename = 'data/feature_dicts/wn_pos_tag_seqs_' + str(prompt_id) + '.pkl'

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            pos_tag_seqs = pickle.load(f)
    else:
        pos_tag_seqs = {}

    if cross_domain:
        if penn_tree_tags:
            pkl_files = ['data/feature_dicts/pos_tag_seqs_' + str(source_prompt) + '.pkl']
        else:
            pkl_files = ['data/feature_dicts/wn_pos_tag_seqs_' + str(source_prompt) + '.pkl']
        pkl_objs = [pos_tag_seqs]
        load_source_pkl(pkl_files, pkl_objs)

    train_out = extract_pos_tag_seq(train_text, train_ids, pos_tag_seqs, penn_tree_tags)
    dev_out = extract_pos_tag_seq(dev_text, dev_ids, pos_tag_seqs, penn_tree_tags)
    test_out = extract_pos_tag_seq(test_text, test_ids, pos_tag_seqs, penn_tree_tags)

    train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, max_sentlen)
    dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, max_sentlen)
    test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, max_sentlen)

    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            pickle.dump(pos_tag_seqs, f)

    return train_padded, dev_padded, test_padded


def tag_modals_word_seq(text):
    output = []
    for sent in text:
        tagged_sentence = pos_tag(sent)
        tag_list = ['MD' if tag == 'MD' else '0' for word, tag in tagged_sentence]
        new_sent = [modal_tagset(tag) for tag in tag_list]
        output.append(new_sent)
    return output


def extract_modals(texts, ids, modals):
    outputs = []

    for text, text_id in zip(texts, ids):
        if text_id in modals:
            output = modals[text_id]
        else:
            output = tag_modals_word_seq(text)
            modals[text_id] = output
        outputs.append(output)
    return outputs


def modal_word(train_text, dev_text, test_text, train_ids, dev_ids, test_ids, prompt_id, max_sentnum, max_sentlen, cross_domain=False, source_prompt=-1):
    filename = 'data/feature_dicts/modals_' + str(prompt_id) + '.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            modals = pickle.load(f)
    else:
        modals = {}

    if cross_domain:
        pkl_files = ['data/feature_dicts/modals_' + str(source_prompt) + '.pkl']
        pkl_objs = [modals]
        load_source_pkl(pkl_files, pkl_objs)

    train_out = extract_modals(train_text, train_ids, modals)
    dev_out = extract_modals(dev_text, dev_ids, modals)
    test_out = extract_modals(test_text, test_ids, modals)

    train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, max_sentlen)
    dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, max_sentlen)
    test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, max_sentlen)

    if not os.path.exists(filename):
        if not cross_domain:
            with open(filename, 'wb') as f:
                pickle.dump(modals, f)

    return train_padded, dev_padded, test_padded


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return ''


def tag_sentiment_word_seq(text, lemmatizer):
    output = []
    for sent in text:
        new_sent = []
        tagged_sentence = pos_tag(sent)
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                new_sent.append([0.0, 1.0, 0.0])
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                new_sent.append([0.0, 1.0, 0.0])
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                new_sent.append([0.0, 1.0, 0.0])
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            new_sent.append([swn_synset.pos_score(), swn_synset.obj_score(), swn_synset.neg_score()])
        output.append(new_sent)
    return output


def extract_sentiment_word(texts, ids, lemmatizer, word_sentiments):
    outputs = []

    for text, text_id in zip(texts, ids):
        if text_id in word_sentiments:
            output = word_sentiments[text_id]
        else:
            output = tag_sentiment_word_seq(text, lemmatizer)
            word_sentiments[text_id] = output
        outputs.append(output)
    return outputs


def sentiment_word(train_text, dev_text, test_text, train_ids, dev_ids, test_ids, prompt_id, max_sentnum, max_sentlen, cross_domain=False, source_prompt=-1):
    filename = 'data/feature_dicts/word_sentiments_' + str(prompt_id) + '.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            word_sentiments = pickle.load(f)
    else:
        word_sentiments = {}

    if cross_domain:
        pkl_files = ['data/feature_dicts/word_sentiments_' + str(source_prompt) + '.pkl']
        pkl_objs = [word_sentiments]
        load_source_pkl(pkl_files, pkl_objs)

    lemmatizer = WordNetLemmatizer()

    train_out = extract_sentiment_word(train_text, train_ids, lemmatizer, word_sentiments)
    dev_out = extract_sentiment_word(dev_text, dev_ids, lemmatizer, word_sentiments)
    test_out = extract_sentiment_word(test_text, test_ids, lemmatizer, word_sentiments)

    train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, max_sentlen, 3, np.float)
    dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, max_sentlen, 3, np.float)
    test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, max_sentlen, 3, np.float)

    if not os.path.exists(filename):
        if not cross_domain:
            with open(filename, 'wb') as f:
                pickle.dump(word_sentiments, f)

    return train_padded, dev_padded, test_padded


def extract_cat_dis_word(parser, texts, ids, parse_trees, discourse_parse_trees):
    # since all texts are pre tokenized, so there is no need to use other fancy tokenizer
    corenlp_properties = {
        'tokenize.language': 'Whitespace'
    }

    # replace brackets because brackets introduce error when construct tree.
    replace_dict = {
        '(': '-LRB-',
        ')': '-RRB-'
    }

    cnt = 0
    outputs = []
    for text, text_id in zip(texts, ids):
        cnt = cnt + 1
        ori_sents = []
        output = []
        if text_id in parse_trees and text_id in discourse_parse_trees:
            parsed_trees = parse_trees[text_id]
            for sent in parsed_trees:
                ori_sents.append(sent.leaves())

            discourse_parsed_trees = discourse_parse_trees[text_id]
        else:
            with open('features/addDiscourse/tmp.txt', 'w') as f:
                if text_id in parse_trees:
                    parsed_trees = parse_trees[text_id]
                else:
                    parsed_trees = []
                    for sent in text:
                        revised_sent = [replace_dict[word] if word in replace_dict else word for word in sent]
                        parsed_sent = next(parser.parse(revised_sent, properties=corenlp_properties))
                        parsed_trees.append(parsed_sent)
                    parse_trees[text_id] = parsed_trees
                    #parsed_trees = parser.parse_sents(revised_text, properties=corenlp_properties)

                for sent_tree in parsed_trees:
                    ori_sents.append(sent_tree.leaves())
                    f.write(' '.join(str(sent_tree).split()) + '\n')

            args_str = "perl features/addDiscourse/addDiscourse.pl --parses features/addDiscourse/tmp.txt"
            args = shlex.split(args_str)
            p = subprocess.Popen(args, stdout=subprocess.PIPE)
            discourse_plain_trees = p.communicate()[0].decode('utf-8')
            discourse_parsed_trees = []
            for discourse_plain_tree in discourse_plain_trees.strip().split('\n'):
                discourse_parsed_trees.append(nltk.tree.Tree.fromstring(discourse_plain_tree))
            discourse_parse_trees[text_id] = discourse_parsed_trees

        for ori_sent, discourse_tree in zip(ori_sents, discourse_parsed_trees):
            new_sent = discourse_tree.leaves()
            out = []
            for i in range(len(ori_sent)):
                if ori_sent[i] == new_sent[i]:
                    out.append(discourse_tagset['0'])
                else:
                    elements = new_sent[i].split('#')
                    out.append(discourse_tagset[elements[-1]])
            output.append(out)
        outputs.append(output)
    return outputs


def cat_dis_word(stanford_path, train_text, dev_text, test_text, train_ids, dev_ids, test_ids, prompt_id, max_sentnum, max_sentlen, cross_domain=False, source_prompt=-1):
    filename_1 = 'data/feature_dicts/parse_trees_' + str(prompt_id) + '.pkl'
    filename_2 = 'data/feature_dicts/discourse_parse_trees_' + str(prompt_id) + '.pkl'
    if os.path.exists(filename_1):
        with open(filename_1, 'rb') as f:
            parse_trees = pickle.load(f)
    else:
        parse_trees = {}

    if os.path.exists(filename_2):
        with open(filename_2, 'rb') as f:
            discourse_parse_trees = pickle.load(f)
    else:
        discourse_parse_trees = {}

    if cross_domain:

        pkl_files = [
            'data/feature_dicts/parse_trees_' + str(source_prompt) + '.pkl',
            'data/feature_dicts/discourse_parse_trees_' + str(source_prompt) + '.pkl'
        ]
        pkl_objs = [
            parse_trees,
            discourse_parse_trees
        ]
        load_source_pkl(pkl_files, pkl_objs)

    # if there is no cached parsed tree
    if len(parse_trees) == 0:

        # Create the server
        server = CoreNLPServer(
            path_to_jar=os.path.join(stanford_path, 'stanford-corenlp-3.9.2.jar'),
            path_to_models_jar=os.path.join(stanford_path, 'stanford-corenlp-3.9.2-models.jar'),
            corenlp_options=['-timeout', '15000']
        )

        # Start the server in the background
        server.start()
        stop_server = True
        parser = CoreNLPParser()
    else:
        stop_server = False
        parser = None

    train_out = extract_cat_dis_word(parser, train_text, train_ids, parse_trees, discourse_parse_trees)
    dev_out = extract_cat_dis_word(parser, dev_text, dev_ids, parse_trees, discourse_parse_trees)
    test_out = extract_cat_dis_word(parser, test_text, test_ids, parse_trees, discourse_parse_trees)

    train_padded = utils.padding_sentence_sequences_without_mask(train_out, max_sentnum, max_sentlen)
    dev_padded = utils.padding_sentence_sequences_without_mask(dev_out, max_sentnum, max_sentlen)
    test_padded = utils.padding_sentence_sequences_without_mask(test_out, max_sentnum, max_sentlen)

    if stop_server:
        if not cross_domain:
            with open(filename_1, 'wb') as f:
                pickle.dump(parse_trees, f)
            with open(filename_2, 'wb') as f:
                pickle.dump(discourse_parse_trees, f)
        server.stop()

    return train_padded, dev_padded, test_padded
