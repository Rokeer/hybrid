#!/usr/bin/env python

## Script to pre-process ASAP dataset (training_set_rel3.tsv) based on the essay IDs

import argparse
import codecs
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', dest='input_file', default='training_set_rel3.tsv', help='Input TSV file')
args = parser.parse_args()

def extract_based_on_ids(dataset, id_file):
    lines = []
    with open(id_file) as f:
        for line in f:
            id = line.strip()
            try:
                lines.append(dataset[id])
            except:
                print(sys.stderr, 'ERROR: Invalid ID %s in %s' % (id, id_file))
    return lines

def create_dataset(lines, output_fname):
    f_write = open(output_fname, 'w')
    f_write.write(dataset['header'])
    # Colin think we should print a new line after the header
    f_write.write("\r\n")
    for line in lines:
        f_write.write(line)
        f_write.write("\r\n")

def collect_dataset(input_file):
    dataset = dict()
    lcount = 0
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            lines = line.splitlines()
            for line in lines:
                lcount += 1
                if lcount == 1:
                    dataset['header'] = line
                    continue
                parts = line.split('\t')
                assert len(parts) >= 6, 'ERROR: ' + line
                dataset[parts[0]] = line
    return dataset

dataset = collect_dataset(args.input_file)
for fold_idx in range(0, 5):
    for dataset_type in ['dev', 'test', 'train']:
        lines = extract_based_on_ids(dataset, 'fold_%d/%s_ids.txt' % (fold_idx, dataset_type))
        create_dataset(lines, 'fold_%d/%s.tsv' % (fold_idx, dataset_type))