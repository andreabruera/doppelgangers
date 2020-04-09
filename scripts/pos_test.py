import os
import numpy
import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager
import scipy
import argparse
import collections
import pickle
import nonce2vec 
import math
import sys
import pickle
import torch
import collections
import re
import logging
import dill
import matplotlib._color_data as mcd

from matplotlib import pyplot
from collections import defaultdict
from scipy import stats
from scipy.stats.morestats import wilcoxon
from re import sub
from nonce2vec.utils.novels_utilities import *
from nonce2vec.utils.count_based_models_utils import cosine_similarity, normalise
from numpy import dot,sqrt,sum,linalg
from math import sqrt
from torch import Tensor
from collections import defaultdict

def wilcoxon_results(x,y):
    
    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]
    try:
        z, p_value = wilcoxon(x,y)
        effect_size = abs(z/math.sqrt(length))
    except (ValueError, FloatingPointError):
        p_value, effect_size = 'nan', 'nan'
    return p_value, effect_size

numpy.seterr(all='raise')

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser=argparse.ArgumentParser()
parser.add_argument('--folder', required=True, type = str, help = 'Specify the name of the folder where the pickles are contained')
parser.add_argument('--make_plots', required=False, action='store_true')
args = parser.parse_args()

cwd = os.getcwd()
big_folder = '{}/{}'.format(cwd, args.folder)
training_types = ['proper_names_matched', 'common_nouns_unmatched', 'common_nouns_matched']
tests = ['doppelganger_test', 'quality_test'] 
setups = [[tr, te] for tr in training_types for te in tests]
all_results = defaultdict(dict)
#pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ', 'VERB']
pos_list = ['ADJ', 'ADV', 'CCONJ', 'DET', 'NOUN', 'PRON', 'PROPN', 'VERB']
window_sizes = ['2', '5', '7', '0']
length = [k for k in range(len(pos_list))]

per_novel_results = defaultdict(dict)

for setup in setups:
    
    logging.info('Currently evaluating performance on setup: {}'.format(setup))

    training = setup[0]
    test = setup[1]
    setup_key = '{}_{}'.format(test, training)
    current_results = defaultdict(list)
    aggregated_results = defaultdict(list)

    for novel in os.listdir(big_folder):
        current_novel = defaultdict(list)
        novel_folder = '{}/{}'.format(big_folder, novel)
        novel_number = re.sub('_no_header.txt', '', [filename for filename in os.listdir('{}/novel'.format(novel_folder)) if 'no_header' in filename][0])

        test_folder = '{}/{}/{}'.format(novel_folder, training, test)
        try:
            current_pickle = pickle.load(open('{}/{}.pickle'.format(test_folder, novel_number), 'rb'))
            for k, v in current_pickle.items():
                window_size = k[-1]
                current_results[window_size].append(v)
                current_novel[window_size].append(v)
            for w in window_sizes:
                aggregated_results[w].append([numpy.median([v[i] for v in current_novel[w]]) for i in range(len(pos_list))])
        except FileNotFoundError:
            logging.info('Could not find the file for {}...'.format(novel))
    all_results[setup_key] = current_results
    per_novel_results[setup_key] = aggregated_results

for test in tests:
    for window_size in window_sizes:
        proper_name_setup = '{}_proper_names_matched'.format(test)
        proper_name_list = per_novel_results[proper_name_setup][window_size]
        common_nouns_setup = '{}_common_nouns_matched'.format(test)
        common_nouns_list = per_novel_results[common_nouns_setup][window_size]
        for i in length:
            p, e = wilcoxon_results([k[i] for k in proper_name_list], [k[i] for k in common_nouns_list])
            os.makedirs('pos_plots/{}'.format(test), exist_ok = True)
            with open('pos_plots/{}/window_{}_significance_results.txt'.format(test, window_size), 'a') as o:
                o.write('Significance results for {}:\n\nP-value:\t{}\nEffect size\t{}\n\nMedian for proper names:\t{}\nMedian for common nouns:\t{}\n\n\n'.format(pos_list[i], p, e, numpy.nanmedian([k[i] for k in proper_name_list]), numpy.nanmedian([k[i] for k in common_nouns_list])))

golden = mcd.CSS4_COLORS['goldenrod']
teal = mcd.CSS4_COLORS['steelblue']

median_results = defaultdict(dict)

for key, dictionary in all_results.items():
    test_results = defaultdict(list)
    for window_size, vectors in dictionary.items():
        test_results[window_size] = [[numpy.nanmedian([v[i] for v in vectors]), numpy.nanstd([v[i] for v in vectors])] for i in range(len(pos_list))]
    median_results[key] = test_results

for test in tests:
    for window_size in window_sizes:
        for key, dictionary in median_results.items():
            if test in key:
                medians = dictionary[window_size]
                y_values = [k[0] for k in medians]
                stds = [k[1] for k in medians]
                if 'proper' in key:
                    pyplot.errorbar([k-0.15 for k in length], y_values, yerr = stds, ecolor='black', elinewidth=0.2, capsize=1, fmt = 'h', label = 'proper names', color = golden, alpha = 0.5)
                elif 'common_nouns_matched' in key:
                    pyplot.errorbar([k+0.15 for k in length], y_values, yerr = stds, ecolor='black', elinewidth=0.2, capsize=1, fmt = 's', label = 'common nouns', color = teal, alpha = 0.5)
        pyplot.ylabel('Median normalized frequency')
        pyplot.xticks(length, pos_list, rotation=45)
        pyplot.legend()
        pyplot.savefig('pos_plots/{}/window_{}.png'.format(test, window_size))
        pyplot.clf()
