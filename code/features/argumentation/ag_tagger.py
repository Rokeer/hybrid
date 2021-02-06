# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   4/26/20 7:50 PM

import requests
import json


class Targer():
    def __init__(
            self,
            model='pe_dep'
    ):
        self.header = {
            'accept': 'application/json',
            'Content-Type': 'text/plain'
        }
        self.model = model

    def _send_request(self, url, post_obj):
        result = requests.post(url, data=post_obj.encode('utf-8'), headers=self.header)
        return result.text

    def _decode_result(self, result):
        json_obj = json.loads(result)
        tokens = []
        labels = []
        probs = []
        for sent in json_obj:
            token = []
            label = []
            prob = []
            for word in sent:
                token.append(word['token'])
                label.append(word['label'])
                if 'prob' in word:
                    prob.append(word['prob'])
                else:
                    prob.append('1.0')
            tokens.append(token)
            labels.append(label)
            probs.append(prob)
        return tokens, labels, probs

    '''
    Get api (IBM model, fasttext embeddings)
    '''

    def _classify_ibm_fasttext(self):
        return 'http://ltdemos.informatik.uni-hamburg.de/arg-api//classifyIBMfasttext'

    '''
    Get api (Essays model, dependency based embeddings)
    '''

    def _classify_pe_dep(self):
        return 'http://ltdemos.informatik.uni-hamburg.de/arg-api//classifyPEdep'

    '''
    Get api (Essays model, fasttext embeddings)
    '''

    def _classify_pe_fasttext(self):
        return 'http://ltdemos.informatik.uni-hamburg.de/arg-api//classifyPEfasttext'

    '''
    Get api (Essays model, glove embeddings)
    '''

    def _classify_pe_glove(self):
        return 'http://ltdemos.informatik.uni-hamburg.de/arg-api//classifyPEglove'

    '''
    Get api (WebD model, dependency based embeddings)
    '''

    def _classify_wd_dep(self):
        return 'http://ltdemos.informatik.uni-hamburg.de/arg-api//classifyWDdep'

    '''
    Get api (WebD model, fasttext embeddings)
    '''

    def _classify_wd_fasttext(self):
        return 'http://ltdemos.informatik.uni-hamburg.de/arg-api//classifyWDfasttext'

    '''
    Get api (WebD model, glove embeddings)
    '''

    def _classify_wd_glove(self):
        return 'http://ltdemos.informatik.uni-hamburg.de/arg-api//classifyWDglove'

    '''
    Classifies input text to argument structure with defined model
    Parameters
    ----------
    model : string
        Name of the model. Select from [ibm_fasttext, pe_dep, pe_fasttext, pe_glove, wd_dep, wd_fasttext, wd_glove]

    text : string
        Input Text.

    Returns
    -------
    tokens : list
        A list of sentences, each sentence is a list of tokens.

    labels : list
        A list of sentences, each sentence is a list of labels.

    probs : list
        A list of sentences, each sentence is a list of probabilities.
    '''

    def classify(self, text):
        if self.model == 'ibm_fasttext':
            url = self._classify_ibm_fasttext()
        elif self.model == 'pe_dep':
            url = self._classify_pe_dep()
        elif self.model == 'pe_fasttext':
            url = self._classify_pe_fasttext()
        elif self.model == 'pe_glove':
            url = self._classify_pe_glove()
        elif self.model == 'wd_dep':
            url = self._classify_wd_dep()
        elif self.model == 'wd_fasttext':
            url = self._classify_wd_fasttext()
        elif self.model == 'wd_glove':
            url = self._classify_wd_glove()
        else:
            raise NotImplementedError
        attempts = 0
        tokens, labels, probs = [], [], []
        while(True):
            try:
                tokens, labels, probs = self._decode_result(self._send_request(url, text))
                break
            except:
                print("Something Wrong, Attempt: " + str(attempts))
                attempts = attempts + 1

        return tokens, labels, probs

    '''
    Change model
    Parameters
    ----------
    model : string
        Name of the model. Select from [ibm_fasttext, pe_dep, pe_fasttext, pe_glove, wd_dep, wd_fasttext, wd_glove]
    '''

    def change_model(self, model):
        self.model = model

# Possible model [ibm_fasttext, pe_dep, pe_fasttext, pe_glove, wd_dep, wd_fasttext, wd_glove]
# tagger = Targer(model='pe_dep')
# text = "Quebecan independence is justified. In the special episode in Japan], his system is restored by a doctor who wishes to use his independence for her selfish reasons."
# tokens, labels, probs = tagger.classify(text)
# print(labels)