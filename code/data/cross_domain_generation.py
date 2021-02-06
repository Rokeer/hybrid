# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   6/17/20 4:11 PM


import codecs
import pandas as pd
import json

seed = 42

for fold in range(5):
    file_path = 'fold_' + str(fold) + '/train.tsv'

    ids = []
    prompts = []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            ids.append(essay_id)
            prompts.append(essay_set)
    df = pd.DataFrame(data={'id': ids, 'prompt': prompts})

    output = {}

    for prompt in range(1, 11):
        p_df = df[df['prompt'] == prompt]
        s_df = p_df.sample(frac=1, random_state=seed * fold + prompt)
        assert len(p_df) == len(s_df)
        output[prompt] = s_df['id'].to_list()

    output_file = 'fold_' + str(fold) + '/shuffled_train.json'

    with open(output_file, 'w') as f:
        json.dump(output, f)
