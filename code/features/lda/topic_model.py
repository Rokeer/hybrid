# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   4/21/20 4:54 PM

# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   9/17/19 1:22 PM

from __future__ import absolute_import, division, print_function, unicode_literals
from gensim.models import HdpModel
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.models import CoherenceModel
import nltk
# python -m spacy download en
# python -m nltk.downloader all
import spacy
import string
import gensim


import utils
import numpy as np

from sklearn import preprocessing
from reader import is_number

# NLTK Stop words
from nltk.corpus import stopwords
logger = utils.get_logger("Topic Model...")

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
nlp = spacy.load('en', disable=['parser', 'ner'])


class TopicModel():
    def __init__(
            self,
            model_mode,
            infer_doc_level,
            remove_stopwords=True,
            deacc=True,
            replace_num=True,
            lemmatize=True,
            lda_len=100,
            normalize=False
    ):
        self.model_mode = model_mode
        self.infer_doc_level = infer_doc_level
        self.remove_stopwords = remove_stopwords
        self.deacc = deacc
        self.replace_num = replace_num
        self.lemmatize = lemmatize
        self.num_topics = lda_len
        self.normalize = normalize

        self.topic_model = None
        self.index = None
        self.id2word = None

    def _rm_stopwords(self, essays):
        return [[[word for word in sent if word not in stop_words] for sent in essay] for essay in essays]

    def _rm_accent(self, essays):
        return [[[word for word in sent if word not in string.punctuation] for sent in essay] for essay in essays]

    def _rp_num(self, essays):
        return [[['<num>' if is_number(word) else word for word in sent] for sent in essay] for essay in essays]
    """
        If you received an error. Pleas run the following command to download the model
        python -m spacy download en
    """
    def _lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def _essay_preprocessing(self, essays, training=True):
        if self.remove_stopwords:
            essays = self._rm_stopwords(essays)
        if self.deacc:
            essays = self._rm_accent(essays)
        if self.replace_num:
            essays = self._rp_num(essays)
        if self.lemmatize:
            essays = [self._lemmatization(essay) for essay in essays]
        texts = []

        if training:
            for essay in essays:
                text = []
                for sent in essay:
                    text.extend(sent)
                texts.append(text)
        else:
            if self.infer_doc_level:
                for essay in essays:
                    text = []
                    for sent in essay:
                        text.extend(sent)
                    texts.append(text)
            else:
                texts = essays
        return texts

    def train_lda(self, corpus, lda_len):
        self.topic_model = LdaModel(
            corpus=corpus,
            id2word=self.id2word,
            random_state=42,
            num_topics=lda_len,
            # update_every=1,
            # chunksize=2000,
            # passes=10,
            # iterations=100,
            # alpha='auto',
            minimum_probability=0.0
        )
        logger.info('Topic model have %d topics.' % lda_len)

    def train_lda_coh(self, corpus, text):

        max_cv = -1
        for lda_len in range(1, 17):
            tmp_topic_model = LdaModel(
                corpus=corpus,
                id2word=self.id2word,
                random_state=42,
                num_topics=lda_len,
                # update_every=1,
                # chunksize=2000,
                # passes=10,
                # iterations=100,
                # alpha='auto',
                minimum_probability=0.0
            )

            coherence_model_lda = CoherenceModel(model=tmp_topic_model, texts=text, dictionary=self.id2word, coherence='c_v')
            cv = coherence_model_lda.get_coherence()
            logger.info('Topic model have %d topics, c_v is %s.' % (lda_len, cv))
            if cv > max_cv:
                logger.info('Find better topic model')
                self.topic_model = tmp_topic_model
                self.num_topics = lda_len
                max_cv = cv

        logger.info('Topic model have %d topics.' % self.num_topics)

    def train_mallet(self, corpus, lda_len):
        mallet_path = 'features/lda/mallet-2.0.6/bin/mallet'
        self.topic_model = LdaMallet(
            mallet_path,
            corpus=corpus,
            num_topics=lda_len,
            id2word=self.id2word,
            random_seed=42
        )
        logger.info('Topic model have %d topics.' % lda_len)

    def init_model(self, essays):
        logger.info('Start text preprocessing...')
        texts = self._essay_preprocessing(essays)
        if self.id2word is None:
            self.id2word = gensim.corpora.Dictionary(texts)
            self.id2word.filter_extremes()
            # special_tokens = {'<pad>': 0, '<unk>': 1}
            special_tokens = {'<pad>': 0}
            self.id2word.patch_with_special_tokens(special_tokens)
            assert self.id2word[0] == '<pad>'
            # assert self.id2word[1] == '<unk>'
        corpus = [self.id2word.doc2bow(text) for text in texts]

        if self.model_mode == 'lda':
            logger.info('Start LDA topic model training...')
            self.train_lda(corpus, self.num_topics)
        elif self.model_mode == 'mallet':
            logger.info('Start Mallet LDA topic model training...')
            self.train_mallet(corpus, self.num_topics)
        elif self.model_mode == 'lda_coh':
            logger.info('Start LDA topic model training... Select best LDA model using coherence score')
            self.train_lda_coh(corpus, texts)

        else:
            raise NotImplementedError

    def get_preprocessed_essays(self, datasets):
        processed_datasets = []
        for dataset in datasets:
            processed_datasets.append(self._essay_preprocessing(dataset, False))
        convert_datasets = [[[self.id2word.doc2idx(sent, 1) for sent in essay] for essay in dataset] for dataset in processed_datasets]
        max_sent_len = 0
        max_sent_num = 0

        for dataset in convert_datasets:
            for essay in dataset:
                if len(essay) > max_sent_num:
                    max_sent_num = len(essay)
                for sent in essay:
                    if len(sent) > max_sent_len:
                        max_sent_len = len(sent)

        return convert_datasets[0], convert_datasets[1], convert_datasets[2], max_sent_num, max_sent_len

    def get_topic_emb_table(self):
        word2topics = self.topic_model.get_topics().transpose()
        return [word2topics]

    def get_lda_topics(self, texts):
        if self.infer_doc_level:
            corpus = [self.id2word.doc2bow(text) for text in texts]
            predicted = self.topic_model[corpus]
            predicted = [[topic[1] for topic in essay] for essay in predicted]
        else:
            corpus = [[self.id2word.doc2bow(sent) for sent in essay] for essay in texts]
            predicted = [self.topic_model[essay] for essay in corpus]
            predicted = [[[topic[1] for topic in sent] for sent in essay] for essay in predicted]
            if self.normalize:
                predicted = [preprocessing.scale(essay, axis=1) for essay in predicted]

        return predicted

    def get_sent_topics(self, datasets):
        processed_datasets = []
        for dataset in datasets:
            texts = self._essay_preprocessing(dataset, False)
            if self.model_mode == 'lda':
                processed_datasets.append(self.get_lda_topics(texts))
            else:
                raise NotImplementedError

        max_sent_num = 0

        for dataset in processed_datasets:
            for essay in dataset:
                if len(essay) > max_sent_num:
                    max_sent_num = len(essay)
                for sent in essay:
                    if len(sent) != self.num_topics:
                        logger.error('Number of topics is not' + str(self.num_topics))
                        raise ValueError

        return processed_datasets[0], processed_datasets[1], processed_datasets[2], max_sent_num

    def get_doc_topics(self, datasets):
        processed_datasets = []
        for dataset in datasets:
            texts = self._essay_preprocessing(dataset, False)
            if self.model_mode == 'lda':
                processed_datasets.append(self.get_lda_topics(texts))
            elif self.model_mode == 'lda_coh':
                processed_datasets.append(self.get_lda_topics(texts))
            else:
                raise NotImplementedError

        for dataset in processed_datasets:
            for essay in dataset:
                if len(essay) != self.num_topics:
                    logger.error('Number of topics is not' + str(self.num_topics))
                    raise ValueError

        return processed_datasets
