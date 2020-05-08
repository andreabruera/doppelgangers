import os
import numpy
import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager
import scipy
import argparse
import collections
import pickle
#import nonce2vec 
import math
import sys
import pickle
import torch
import collections
import re
import logging
import dill
import matplotlib._color_data as mcd

from data.mybloodyplots.mybloodyplots import MyBloodyPlots
from matplotlib import pyplot
from collections import defaultdict
from scipy import stats
from scipy.stats.morestats import wilcoxon
from re import sub
#from nonce2vec.utils.novels_utilities import *
#from nonce2vec.utils.count_based_models_utils import cosine_similarity, normalise
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
parser.add_argument('--folder', required=True, type = str, help = 'Specify the absolute path to the folder where the pickles are contained')
parser.add_argument('--make_plots', action='store_true', default=False, help='Indicates whether to plot the results or not')
parser.add_argument('--write_to_file', action='store_true', default=False, help='Indicates whether to write the results to file or not')
args = parser.parse_args()

# Preparing some utilities

#training_types = ['proper_names_matched', 'common_nouns_unmatched', 'common_nouns_matched']
training_types = ['proper_names_matched', 'common_nouns_unmatched']
tests = ['doppelganger_test', 'quality_test'] 
setups = [(tr, te) for tr in training_types for te in tests]
all_results = defaultdict(dict)
#pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'SCONJ', 'VERB']
pos_list = ['ADJ', 'ADV', 'CCONJ', 'DET', 'NOUN', 'PRON', 'PROPN', 'VERB']
window_sizes = ['2', '5', '7', '0']
length = [k for k in range(len(pos_list))]
output_folder = 'pos_plots'
os.makedirs(output_folder, exist_ok=True)

per_novel_results = defaultdict(dict)

# Collecting the evaluations for each setup

for setup_key in setups:
    
    logging.info('Currently evaluating performance on setup: {}'.format(setup_key))

    training = setup_key[0]
    test = setup_key[1]
    #setup_key = '{}_{}'.format(test, training)
    
    current_results = defaultdict(list)
    aggregated_results = defaultdict(list)

    # Collecting the results for each novel

    for novel in os.listdir(args.folder):
        current_novel = defaultdict(list)
        novel_folder = os.path.join(args.folder, novel)
        novel_number = re.sub('_no_header.txt', '', [filename for filename in os.listdir('{}/novel'.format(novel_folder)) if 'no_header' in filename][0])

        test_folder = os.path.join(novel_folder, training, test)
        try:
            current_pickle = pickle.load(open('{}.pickle'.format(os.path.join(test_folder, novel_number)), 'rb'))
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

# Now calculating the statistical significance of the differences among POS, only for the novels

test = 'doppelganger_test'
for window_size in window_sizes:
    #proper_name_setup = '{}_proper_names_matched'.format(test)
    proper_name_list = per_novel_results[('proper_names_matched', test)][window_size]
    #common_nouns_setup = '{}_common_nouns_unmatched'.format(test)
    common_nouns_list = per_novel_results[('common_nouns_unmatched', test)][window_size]
    for i in length:
        p, e = wilcoxon_results([k[i] for k in proper_name_list], [k[i] for k in common_nouns_list])
        if args.write_to_file:
            with open(os.path.join(output_folder, 'window_{}_significance_results.txt'.format(window_size)), 'a') as o:
                o.write('Significance results for {}:\n\nP-value:\t{}\nEffect size\t{}\n\nMedian for proper names:\t{}\nMedian for common nouns:\t{}\n\n\n'.format(pos_list[i], p, e, numpy.nanmedian([k[i] for k in proper_name_list]), numpy.nanmedian([k[i] for k in common_nouns_list])))
        #print('Significance results for {}:\n\nP-value:\t{}\nEffect size\t{}\n\nMedian for proper names:\t{}\nMedian for common nouns:\t{}\n\n\n'.format(pos_list[i], p, e, numpy.nanmedian([k[i] for k in proper_name_list]), numpy.nanmedian([k[i] for k in common_nouns_list])))


golden = mcd.CSS4_COLORS['goldenrod']
teal = mcd.CSS4_COLORS['steelblue']

# Collecting the median results for plotting

median_results = defaultdict(dict)
for key, dictionary in all_results.items():
    test_results = defaultdict(list)
    for window_size, vectors in dictionary.items():
        test_results[window_size] = [[numpy.nanmedian([v[i] for v in vectors]), numpy.nanstd([v[i] for v in vectors])] for i in range(len(pos_list))]
    median_results[key] = test_results

# Actually plotting
labels = ['Common nouns', 'Proper names']
colors = [teal, golden]

# Making one plot per window size, only for the novels data
for window_size in window_sizes:

    common_data = median_results[('common_nouns_unmatched', test)][window_size]
    proper_data = median_results[('proper_names_matched', test)][window_size]
    x_variables = pos_list

    errorbar_plots = MyBloodyPlots(output_folder=output_folder, font_folder='/import/cogsci/andrea/fonts', x_variables=x_variables, y_variables=[common_data, proper_data], x_axis='', y_axis='Median normalized frequency', labels=labels, colors=colors, identifier='window_{}'.format(window_size), title='Window {} - POS analysis for the {}'.format(window_size, re.sub('_', ' ', test).capitalize()), y_ticks=True)
    errorbar_plots.plot_dat(plot_type='errorbar_two_sets')
