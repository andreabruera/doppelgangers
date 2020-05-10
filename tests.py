import os
import numpy
import matplotlib
import matplotlib.cm as cm
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
import tqdm
import matplotlib.cm as cm
import itertools

from data.mybloodyplots.mybloodyplots import MyBloodyPlots
from tqdm import tqdm
from matplotlib import pyplot
from matplotlib import font_manager as font_manager
from collections import defaultdict
from scipy import stats
from scipy.stats.morestats import wilcoxon
from re import sub
from numpy import dot,sqrt,sum,linalg
from math import sqrt
from torch import Tensor
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize as normalise
from numpy.linalg import norm

# Some functions for the tSNE visualization

def tsne_plot_words(title, words, embeddings, colors, filename=None):
    pyplot.figure(figsize=(16, 9))
    for embedding, word in zip(embeddings, words):
        #x = embeddings[:, 0]
        #y = embeddings[:, 1]
        #import pdb; pdb.set_trace()
        pyplot.scatter(embedding[0], embedding[1], c=(colors[word].reshape(1, colors[word].shape[0])), alpha=1, edgecolors='k', s=120)
        pyplot.annotate(word, alpha=1, xy=(embedding[0], embedding[1]), xytext=(10, 7), textcoords='offset points', ha='center', va='bottom', size=12)
    #pyplot.legend(loc=4)
    #pyplot.title(title, fontdict={'fontsize': 24, 'fontweight' : 'bold', 'color' : rcParams['axes.titlecolor'], 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}, pad=10.0)
    pyplot.title(title, fontsize='xx-large', fontweight='bold', pad = 15.0)
    #pyplot.grid(True)
    if filename:
        pyplot.savefig(filename, format='png', dpi=300, bbox_inches='tight')

def cleanup(version, dictionary, mode='doppelganger'):
    name_selection = []
    second_label = 'b' if mode == 'doppelganger' else 'wiki'
    for name, vector in dictionary.items():
        if re.sub('_a$', '_{}'.format(second_label), name) in dictionary.keys() and re.sub('_{}$'.format(second_label), '_a', name) in dictionary.keys():
            if name not in name_selection:
                name_selection.append(name)
    cleaned_up = {name : dictionary[name] for name in name_selection} 
    return cleaned_up

def merge_two_dicts(version, dict_doppelganger, dict_one):
    z = dict_doppelganger.copy()
    z_clean = {k : v for k, v in z.items() if k[-1] == 'a'}
    z_clean.update(dict_one)
        
    return z_clean

def prepare_damn_numpy_arrays(dictionary):
    new_dict = defaultdict(numpy.ndarray)
    for k, v in dictionary.items():
        if (v.shape)[0] == 1:
            new_dict[k] = v.reshape(v.shape[1])
        else:
            new_dict[k] = v
    return(new_dict)


def get_colors_dict(test, proper_names, common_nouns):
    #colors_gen = cm.prism(numpy.linspace(0, 1, (len(names)*2)))
    color_dict = defaultdict(numpy.ndarray)
    collection = {'proper' : [k for k, v in proper_names.items()], 'common' : [k for k, v in common_nouns.items()]}
    for category, content in collection.items():
        if category == 'proper': 
            colors_gen = cm.Wistia(numpy.linspace(0, 1, (len(proper_names.keys()))))
        if category == 'common':
            colors_gen = cm.winter(numpy.linspace(1, 0, (len(common_nouns.keys()))))
        c = 0
        for name in content:
            if test == 'doppelganger':
                label = 'b'
            if test == 'quality':
                label = 'wiki'
            na = re.sub('_a$|_b$|_wiki$', '', name)
            if '{}_a'.format(na) not in color_dict.keys() and '{}_{}'.format(na, label) not in color_dict.keys():
                #print(n)
                c += 1
                color_dict['{}_a'.format(na)] = colors_gen[c]
                color_dict['{}_{}'.format(na, label)] = colors_gen[c]
    return color_dict

# The main RSA calculation function

def calculate_pairwise_comparisons(vectors):
    cosines = []
    for first_name, first_vector in vectors.items():
        for second_name, second_vector in vectors.items():
            if first_name != second_name:
                cosines.append(cosine_similarity(normalise(first_vector), normalise(second_vector)))
    return cosines

# Some general utilities

def normalise(vector):
    norm_vector = norm(vector)
    if norm_vector == 0:
        return vector
    vector = vector / norm_vector
    #print(sum([i*i for i in v]))
    return vector

def cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = numpy.dot(peer_v, query_v)
    den_a = numpy.dot(peer_v, peer_v)
    den_b = numpy.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))

def wilcoxon_results(x,y):
    
    length = min(len(x), len(y))
    x = x[:length]
    y = y[:length]
    z, p_value = wilcoxon(x,y)
    effect_size = abs(z/math.sqrt(length))
    return p_value, effect_size

# The main evaluation function

def evaluation(entity_vectors, folder, entities_limit = 1000):

    ranks = []
    full_entities_list = [re.sub('_a$|_b$|_wiki$', '', entity) for entity in entity_vectors.keys()]
    characters_counter = defaultdict(int)
    for entity in full_entities_list:
        characters_counter[entity] += 1
    current_list_of_entities = [entity for entity in full_entities_list if characters_counter[entity] == 2]

    for good_key, good_vector in entity_vectors.items():
        good_counter = defaultdict(int)
        norm_good_vector = normalise(good_vector)
        entity_name = re.sub('_a$|_b$|_wiki$', '', good_key)
        entity_part = good_key[len(good_key)-1]
        if entity_name in current_list_of_entities and good_counter[entity_part] <= entities_limit:
            good_counter[entity_part] += 1
            simil_list = []
            other_counter = defaultdict(int)
            for other_key, other_vector in entity_vectors.items():
                other_name = re.sub('_a$|_b$|_wiki$', '', other_key)
                other_part = other_key[len(other_key)-1]
                if other_part != entity_part and other_name in current_list_of_entities and other_counter[entity_part] <= entities_limit:
                    other_counter[entity_part] +=1
                    norm_other_vector = normalise(other_vector)
                    simil = cosine_similarity(norm_good_vector, norm_other_vector)
                    simil_list.append([other_key, float(simil)])
            sorted_simil_list = sorted(simil_list, reverse=True, key = lambda s : s[1])

            for rank, sim_tuple in enumerate(sorted_simil_list):
                rank += 1
                current_entity = re.sub('_a$|_b$|_wiki$', '', sim_tuple[0])
                if current_entity == entity_name:
                    ranks.append([good_key, rank])
                    with open('{}/similarities_results.txt'.format(folder), 'a') as s:
                        s.write('Result for the vector: {}, coming from part {} of the book\nRank: {} out of {} characters\nCosine similarity to the co-referring vector: {}\n\n'.format(entity_name, entity_part, rank, len(sorted_simil_list), sim_tuple[1]))
                        for other_rank, other_sim_tuple in enumerate(sorted_simil_list):
                                s.write('{}) {} - {}\n\n'.format(other_rank+1, other_sim_tuple[0], other_sim_tuple[1]))

    if len(ranks) > 3:
        try:
            assert len(ranks) % 2 == 0
        except AssertionError:
            import pdb; pdb.set_trace()
        pass
    else:
        #logging.info('Not enough evaluations for {}!\nCheck the number of characters or common nouns for the current novel'.format(folder))
        pass

    return ranks

# Start of the evaluation script

parser=argparse.ArgumentParser()
parser.add_argument('--data_folder', required=True, type = str, help = 'Specify the absolute path of the data folder where the folders containing the pickles for the POS and the main analyses are contained')
args = parser.parse_args()

# Petty stuff setup

numpy.seterr(all='raise')
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
labels = ['Common nouns', 'Proper names']
golden = mcd.CSS4_COLORS['goldenrod']
teal = mcd.CSS4_COLORS['steelblue']
colors = [teal, golden]
current_font_folder = '/import/cogsci/andrea/fonts'
main_folder = 'results/main_results'
os.makedirs(main_folder, exist_ok=True)

# Setting up the evaluation

cwd = os.getcwd()

main_analyses_folder = os.path.join(args.data_folder, 'main_pickles')
count_model_folder = os.path.join(args.data_folder, 'count_models')
folders = os.listdir(main_analyses_folder)

final_results = defaultdict(dict)
histogram_results = defaultdict(dict)
training_types = ['proper_names_matched', 'common_nouns_unmatched', 'common_nouns_matched']
tests = ['doppelganger_test', 'quality_test'] 
setups = [[tr, te] for tr in training_types for te in tests]

# Starting the evaluation

if 'bert_base_training' in folders and 'bert_large_training'in folders and 'elmo_training' in folders and 'n2v_training' in folders and 'w2v_training' in folders and 'count_training' in folders:
    folders = ['count_training', 'w2v_training', 'n2v_training', 'elmo_training', 'bert_base_training', 'bert_large_training']

for folder in tqdm(folders):

    model = re.sub('_training', '', folder)
    big_folder = os.path.join(main_analyses_folder, folder)
    all_results = defaultdict(dict)
    logging.info('Current model: {}'.format(model))

    # Utilities
    doppelganger_entities_limit = defaultdict(int)
    quality_entities_limit = defaultdict(int)
    if 'count' in folder:
        with open(os.path.join(count_model_folder, 'count_wiki_2/count_wiki_2_cooccurrences.pickle'), 'rb') as word_cooccurrences_file:
            background_vectors_length = max(dill.load(word_cooccurrences_file).keys())
            #logging.info(background_vectors_length)

    # Variables for the correlational analyses

    number_of_characters = defaultdict(dict)
    novel_lengths = defaultdict(int)
    characters_std = defaultdict(float)

    # Collecting results per novel, once per experimental setup

    for setup in setups:
        
        training = setup[0]
        test = setup[1]
        setup_key = '{}_{}'.format(test, training)
        current_results = defaultdict(dict)
        #logging.info('Currently evaluating performance on setup: {}'.format(setup_key))

        for novel in os.listdir(big_folder):

            novel_folder = '{}/{}'.format(big_folder, novel)
            novel_number = re.sub('_no_header.txt', '', [filename for filename in os.listdir('{}/novel'.format(novel_folder)) if 'no_header' in filename][0])

            test_folder = '{}/{}/{}'.format(novel_folder, training, test)
            try:

                # Acquiring the information needed for the correlational analyses for the novel at hand

                if training == 'proper_names_matched':
                    novel_lengths[novel] = len([w for l in open('{}/novel/{}_no_header.txt'.format(novel_folder, novel_number)).readlines() for w in l.split()])
                    characters_std[novel] = numpy.nanstd([int(l.split('\t')[0]) for l in open('{}/characters_list_{}.txt'.format(novel_folder, novel_number)).readlines() if int(l.split('\t')[0]) >= 10])

                # Loading the entity vectors

                current_pickle = pickle.load(open('{}/{}.pickle'.format(test_folder, novel_number), 'rb'))
                if 'count' in folder:
                    entity_dict = defaultdict(numpy.ndarray)
                    for e, v in current_pickle.items():
                        entity_dict[e] = v[0][:background_vectors_length]
                else:
                    entity_dict = current_pickle

                # Preparing the entity vectors for the quality test

                if test == 'quality_test':
                    full_novel_pickle = pickle.load(open('{}/{}/doppelganger_test/{}.pickle'.format(novel_folder, training, novel_number), 'rb'))
                    for e, v in full_novel_pickle.items():
                        if e[-1] == 'a':
                            if 'count' in folder:
                                entity_dict[e] = v[0][:background_vectors_length]
                            else:
                                entity_dict[e] = v

                # Carrying out the evaluation for the doppelganger test

                if test == 'doppelganger_test':
                    if 'proper' not in training:
                        current_results[novel] = evaluation(entity_dict, test_folder, doppelganger_entities_limit[novel])
                    else:
                        current_results[novel] = evaluation(entity_dict, test_folder)
                        doppelganger_entities_limit[novel] = int(len(current_results[novel])/2)
                
                # Carrying out the evaluation for the quality test

                else:
                    if quality_entities_limit[novel] > 0:
                        current_results[novel] = evaluation(entity_dict, test_folder, quality_entities_limit[novel])
                        quality_entities_limit[novel] = min(quality_entities_limit[novel], len(current_results[novel])/2)
                    else:
                        current_results[novel] = evaluation(entity_dict, test_folder)
                        quality_entities_limit[novel] = int(len(current_results[novel])/2)

                assert len(current_results[novel]) % 2 == 0
            except FileNotFoundError:
                logging.info('Could not find the file for {}...'.format(novel))
        all_results[setup_key] = current_results

    model_results = defaultdict(dict)
    model_histogram = defaultdict(list)

    for setup in setups:
        training = setup[0]
        test = setup[1]
        setup_key = '{}_{}'.format(test, training)
        results = all_results[setup_key]
        across_novels_results = defaultdict(list)

        test_short = re.sub('_test', '', test)
        os.makedirs(os.path.join(main_folder, test_short), exist_ok=True)
        with open(os.path.join(main_folder, test_short, '{}_details.txt'.format(re.sub('_training', '', folder))), 'a') as o:
            o.write('{} test\n\n'.format(test_short.capitalize()))

        # Averaging the results within the novels

        for novel, results_list in results.items():
            ranks = [k[1] for k in results_list]
            reciprocal_ranks = [1/k[1] for k in results_list]
            amount_of_entities = len(results_list)/2
            novel_median = numpy.median(ranks)
            novel_average = numpy.average(ranks)
            novel_mrr = numpy.average(reciprocal_ranks)
            novel_std = numpy.std(ranks)
            across_novels_results[novel] = [novel_median, novel_average, novel_mrr, novel_std, amount_of_entities]

            # Writing them down to file

            output_file = os.path.join(main_analyses_folder, folder, novel, training, test, 'final_results.txt')
            with open(output_file, 'w') as o:
                o.write('Test: {}\nResults for: {}\nModel used: {}\n\nWithin-novel median: {}\nWithin-novel average: {}\nWithin-novel MRR: {}\nWithin-novel standard deviation of the scores: {}\nAmount of entities considered: {}'.format(re.sub('_', ' ', test), novel, re.sub('_', ' ', training), novel_median, novel_average, novel_mrr, novel_std, amount_of_entities))

        # Averaging results across the novels

        list_of_medians = [m[0] for n, m in across_novels_results.items()]
        across_median = numpy.nanmedian(list_of_medians)
        across_average = numpy.nanmean([m[1] for n, m in across_novels_results.items()])
        across_average_mrr = numpy.nanmean([m[2] for n, m in across_novels_results.items()])
        across_median_mrr = numpy.nanmedian([m[2] for n, m in across_novels_results.items()])
        across_average_std = numpy.nanmean([m[3] for n, m in across_novels_results.items()])
        across_median_std = numpy.nanmedian([m[3] for n, m in across_novels_results.items()])
        number_of_novels_used = len(list_of_medians)

        # Acquiring the last missing variable for the correlational analyses, the number of characters considered

        list_of_entities = [m[4] for n, m in across_novels_results.items()]
        across_entities = numpy.nanmedian(list_of_entities)

        # Carrying out the correlational analysis
        try:
            number_of_characters_correlation = scipy.stats.spearmanr(list_of_medians, list_of_entities)[0]
            novel_length_correlation = scipy.stats.spearmanr(list_of_medians, [k[1] for k in novel_lengths.items()])[0]
            std_of_characters_correlation = scipy.stats.spearmanr(list_of_medians, [k[1] for k in characters_std.items()])[0]
        except FloatingPointError:
            number_of_characters_correlation = numpy.nan
            novel_length_correlation = numpy.nan
            std_of_characters_correlation = numpy.nan

        # Appending the results for the final plots

        model_results[setup_key] = {'average_rank' : across_average, 'median_rank' : across_median, 'average_mrr' : across_average_mrr, 'median_mrr' : across_median_mrr, 'corr_num_characters' : number_of_characters_correlation, 'corr_novel_length' : novel_length_correlation, 'corr_std_characters_mentions' : std_of_characters_correlation}

        model_histogram[setup_key] = list_of_medians

        # Writing down the results

        with open(os.path.join(main_folder, test_short, '{}_details.txt'.format(re.sub('_training', '', folder))), 'a') as o:
            o.write('Linguistic category considered: {}\n\nAcross-novels median of the medians: {}\nAcross-novels average of the averages: {}\nAcross-novels average of the MRRs: {}\nAcross-novels average standard deviation of the scores: {}\nAcross-novels median of the amount of entities considered: {}\nNumber of novels used: {}\n\nCorrelation with number of characters: {}\nCorrelation with the length of the novels: {}\nCorrelation with the standard deviation of the character\'s frequencies: {}\n'.format(re.sub('_', ' ', training).capitalize(), across_median, across_average, across_average_mrr, across_average_std, across_entities, number_of_novels_used, number_of_characters_correlation, novel_length_correlation, std_of_characters_correlation))

        # Printing the results

        #logging.info('\nTest: {}\nLinguistic category considered: {}\n\nAcross-novels median of the medians: {}\nAcross-novels average of the averages: {}\nAcross-novels average of the MRRs: {}\nAcross-novels average standard deviation of the scores: {}\nAcross-novels median of the amount of entities considered: {}\nNumber of novels used: {}\n\nCorrelation with number of characters: {}\nCorrelation with the length of the novels: {}\nCorrelation with the standard deviation of the character\'s frequencies: {}\n'.format(re.sub('_', ' ', test), re.sub('_', ' ', training), across_median, across_average, across_average_mrr, across_average_std, across_entities, number_of_novels_used, number_of_characters_correlation, novel_length_correlation, std_of_characters_correlation))

    final_results[model] = model_results
    histogram_results[model] = model_histogram
   

# Reorganizing the data for plotting out the final results

# A reminder: model_results[setup_key] = {'average_rank' : across_average, 'median_rank' : across_median, 'average_mrr' : across_average_mrr, 'median_mrr' : across_median_mrr, 'corr_num_characters' : number_of_characters_correlation, 'corr_novel_length' : novel_length_correlation, 'corr_std_characters_mentions' : std_of_characters_correlation}

plottable_results = defaultdict(list)
novel_length = []
number_of_characters = []
std_of_character_mentions = []
models = [(re.sub('_', ' ', m)).capitalize() for m in final_results.keys()]

# Data for the main results and the correlational analyses

for model_name, setup_keys in final_results.items():
    for t in tests:
        t = re.sub('_test', '', t)
        plottable_results[t].append((setup_keys['{}_test_common_nouns_unmatched'.format(t)]['median_rank'], setup_keys['{}_test_proper_names_matched'.format(t)]['median_rank']))
    novel_length.append(setup_keys['doppelganger_test_proper_names_matched']['corr_novel_length'])
    number_of_characters.append(setup_keys['doppelganger_test_proper_names_matched']['corr_num_characters'])
    std_of_character_mentions.append(setup_keys['doppelganger_test_proper_names_matched']['corr_std_characters_mentions'])

# Data for the histogram

plottable_histogram = defaultdict(list)
for model_name, setup_keys in histogram_results.items():
    for t in tests:
        t = re.sub('_test', '', t)
        plottable_histogram[t].append((setup_keys['{}_test_common_nouns_unmatched'.format(t)], setup_keys['{}_test_proper_names_matched'.format(t)]))

# Creating the folders for the plots

corr_folder = 'results/correlations'
hist_folder = 'results/histograms'
os.makedirs(corr_folder, exist_ok=True)
os.makedirs(hist_folder, exist_ok=True)

# Plotting the main results
logging.info('Plotting the main results and the correlational analyses...')

for test, results in plottable_results.items():
    results_plots = MyBloodyPlots(output_folder=os.path.join(main_folder, test), font_folder=current_font_folder, x_variables=models, y_variables=results, x_axis='', y_axis='Median rank', labels=['Common nouns', 'Proper names'], title='Median ranking results for the {} test'.format(test.capitalize()), identifier=test, colors=[teal, golden], y_invert=True, x_ticks=True, y_ticks=True, y_grid=True)
    results_plots.plot_dat('two_lines')

    # Plotting the correlational analyses

    y_ticks = [round(k*.1, 1) for k in range(0, 11)]
    corr_plot = MyBloodyPlots(output_folder=corr_folder, font_folder=current_font_folder, x_variables=models, y_variables=[novel_length, number_of_characters, std_of_character_mentions], x_axis='', y_axis='Spearman correlation', labels=['Novel length', 'Number of characters', 'Std of character mentions'], title='Correlational analysis for the results on proper names', identifier='correlations', colors=['darkorange', 'orchid', 'darkgrey'], x_ticks=True, y_ticks=True, y_lim=(0.0, 1.0), y_grid=True)
    corr_plot.plot_dat('three_bars')

# Plotting the histogram analyses

logging.info('Plotting the histograms...')
    
for test_name, results in plottable_histogram.items():
    hist_folder_per_test = os.path.join(hist_folder, test_name)
    os.makedirs(hist_folder_per_test, exist_ok=True)
    for results_index, variables_tuple in enumerate(results):
        model = models[results_index]
        hist_plots = MyBloodyPlots(output_folder=hist_folder_per_test, font_folder=current_font_folder, x_variables=[], y_variables=variables_tuple, x_axis='Median rank', y_axis='Frequency (N=59)', labels=['Common nouns', 'Proper names'], title='{} model - Histogram of the median ranks for the {} test'.format(model, test_name.capitalize()), identifier='{}_{}'.format(test_name.lower(), re.sub('\s', '_', model).lower()), colors=[teal, golden], y_invert=False, y_ticks=True, x_ticks=True, y_grid=True)
        hist_plots.plot_dat('histogram_two_sets')

        # Comparing the statistical significance across different conditions

        significance_results = []
        try:
            significance_results.append([wilcoxon_results(variables_tuple[0], variables_tuple[1]), numpy.median(variables_tuple[0]), numpy.median(variables_tuple[1])])
        except ValueError:
            significance_results.append([(numpy.nan, numpy.nan), numpy.median(variables_tuple[0]), numpy.median(variables_tuple[1])])
        with open(os.path.join(hist_folder_per_test, 'significance_test_results.txt'), 'a') as o:
            for s in significance_results:
                o.write('{} model\n\n\tCommon nouns - median: {}\n\tProper names - median: {}\n\nP-value: {}\nEffect size: {}\n\n\n'.format(re.sub('_', ' ', model).capitalize(), s[1], s[2], s[0][0], s[0][1]))

# Plotting the POS analysis

logging.info('Plotting the POS analysis plots...')

# Preparing some utilities

all_results = defaultdict(dict)
pos_list = ['ADJ', 'ADV', 'CCONJ', 'DET', 'NOUN', 'PRON', 'PROPN', 'VERB']
window_sizes = ['2', '5', '7', '10']
length = [k for k in range(len(pos_list))]
pos_folder = os.path.join(args.data_folder, 'pos_pickles')
output_pos_folder = 'results/pos'
os.makedirs(output_pos_folder, exist_ok=True)
setups = [(tr, te) for tr in training_types for te in tests]

per_novel_results = defaultdict(dict)

# Collecting the evaluations for each setup

for setup_key in setups:
    
    #logging.info('Currently evaluating performance on setup: {}'.format(setup_key))

    training = setup_key[0]
    test = setup_key[1]
    #setup_key = '{}_{}'.format(test, training)
    
    current_results = defaultdict(list)
    aggregated_results = defaultdict(list)

    # Collecting the results for each novel

    for novel in os.listdir(pos_folder):
        current_novel = defaultdict(list)
        novel_folder = os.path.join(pos_folder, novel)
        novel_number = re.sub('_no_header.txt', '', [filename for filename in os.listdir('{}/novel'.format(novel_folder)) if 'no_header' in filename][0])

        test_folder = os.path.join(novel_folder, training, test)
        try:
            current_pickle = pickle.load(open('{}.pickle'.format(os.path.join(test_folder, novel_number)), 'rb'))
            for k, v in current_pickle.items():
                window_size = re.sub('\D', '', k)
                current_results[window_size].append(v)
                current_novel[window_size].append(v)
            for w in window_sizes:
                aggregated_results[w].append([numpy.median([v[i] for v in current_novel[w]]) for i in range(len(pos_list))])
        except FileNotFoundError:
            #logging.info('Could not find the file for {}...'.format(novel))
            pass
    all_results[setup_key] = current_results
    per_novel_results[setup_key] = aggregated_results

# Collecting the median results for plotting

median_results = defaultdict(dict)
full_results = defaultdict(dict)
for key, dictionary in all_results.items():
    test_median_results = defaultdict(list)
    test_full_results = defaultdict(list)
    
    for window_size, vectors in dictionary.items():
        test_median_results[window_size] = [[numpy.nanmedian([v[i] for v in vectors]), numpy.nanstd([v[i] for v in vectors])] for i in range(len(pos_list))]
        test_full_results[window_size] = [[v[i] for v in vectors] for i in range(len(pos_list))]
    median_results[key] = test_median_results
    full_results[key] = test_full_results

# Actually plotting
test = 'doppelganger_test'

# Making one plot per window size, only for the novels data
for window_size in window_sizes:

    #pos_list = ['ADJ', 'ADV', 'CCONJ', 'DET', 'NOUN', 'PRON', 'PROPN', 'VERB']
    common_data = [k for i, k in enumerate(median_results[('common_nouns_unmatched', test)][window_size]) if i != 2 and i != 6] 
    proper_data = [k for i, k in enumerate(median_results[('proper_names_matched', test)][window_size]) if i != 2 and i != 6]
    x_variables = [k for i, k in enumerate(pos_list) if i != 2 and i != 6]

    errorbar_plots = MyBloodyPlots(output_folder=output_pos_folder, font_folder='/import/cogsci/andrea/fonts', x_variables=x_variables, y_variables=[common_data, proper_data], x_axis='', y_axis='Median normalized frequency', labels=labels, colors=colors, identifier='window_{}'.format(window_size), title='Window {} - POS analysis for the {}'.format(window_size, re.sub('_', ' ', test).capitalize()), y_ticks=True, y_lim=(0.0, 1.0), y_grid=True)
    errorbar_plots.plot_dat(plot_type='errorbar_two_sets')

    # Now calculating the statistical significance of the differences among POS, only for the novels

    common_data = [k for i, k in enumerate(full_results[('common_nouns_unmatched', test)][window_size]) if i != 2 and i != 6] 
    proper_data = [k for i, k in enumerate(full_results[('proper_names_matched', test)][window_size]) if i != 2 and i != 6]
    x_variables = [k for i, k in enumerate(pos_list) if i != 2 and i != 6]
    for pos_index, pos in enumerate(x_variables):
        p, e = wilcoxon_results(proper_data[pos_index], common_data[pos_index])
        with open(os.path.join(output_pos_folder, 'window_{}_significance_results.txt'.format(window_size)), 'a') as o:
            o.write('Significance results for {}:\n\nP-value:\t{}\nEffect size\t{}\n\nMedian for proper names:\t{}\nMedian for common nouns:\t{}\n\n\n'.format(pos, p, e, numpy.nanmedian(proper_data[pos_index]), numpy.nanmedian(common_data[pos_index])))

# Plotting the tSNE plots for A Study in Scarlet

logging.info('Plotting the tSNE plots...')

tests = ['quality', 'doppelganger']
versions = ['bert_base', 'bert_large', 'elmo', 'n2v', 'w2v', 'count']
combinations = []
tSNE_output_folder = 'results/tSNE'
os.makedirs(tSNE_output_folder, exist_ok=True)

for t in tests:
    for v in versions:
        if (t, v) not in combinations:
            combinations.append((t, v))

for combination in combinations:

    test = combination[0]
    version = combination[1]

    os.makedirs(os.path.join(tSNE_output_folder, test), exist_ok=True)
    #v = pickle.load(open('elmo_novels/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/244.pickle', 'rb'))
    pickle_proper = prepare_damn_numpy_arrays(pickle.load(open(os.path.join(main_analyses_folder, '{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/proper_names_matched/{}_test/244.pickle'.format(version, test)), 'rb')))

    pickle_common = prepare_damn_numpy_arrays(pickle.load(open(os.path.join(main_analyses_folder, '{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/common_nouns_unmatched/{}_test/244.pickle'.format(version, test)), 'rb')))

    v_proper = cleanup(version, pickle_proper)
    v_common = cleanup(version, pickle_common)
    
    
    if test == 'quality':
        
        v_proper_prepared = prepare_damn_numpy_arrays(pickle.load(open(os.path.join(main_analyses_folder, '{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/proper_names_matched/doppelganger_test/244.pickle'.format(version)), 'rb')))
        v_proper_merged = merge_two_dicts(version, v_proper_prepared, v_proper)
        v_common_prepared = prepare_damn_numpy_arrays(pickle.load(open(os.path.join(main_analyses_folder, '{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/common_nouns_unmatched/doppelganger_test/244.pickle'.format(version)), 'rb')))
        v_common_merged = merge_two_dicts(version, v_common_prepared, v_common)
        v_proper = cleanup(version, v_proper_merged, mode='wiki')
        v_common = cleanup(version, v_common_merged, mode='wiki')
    if version == 'count':
        minimum_cutoff = min([v.shape[0] for k, v in v_proper.items()] + [v.shape[0] for k, v in v_common.items()])
        proper_copy = v_proper.copy()
        common_copy = v_common.copy()
        v_proper = {k : v[:minimum_cutoff] for k, v in proper_copy.items()}
        v_common = {k : v[:minimum_cutoff] for k, v in common_copy.items()}

    if version == 'bert_large':
        version = 'BERT_large'
    if version == 'bert_base':
        version = 'BERT_base'
    if version == 'elmo':
        version = 'ELMO'
    if version == 'n2v':
        version = 'Nonce2Vec'
    if version == 'w2v':
        version = 'Word2Vec'
    if version == 'count':
        version = 'Count'

    
    colors_dict = get_colors_dict(test, v_proper, v_common)

    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = tsne_model_en_2d.fit_transform([v for k, v in v_proper.items()] + [v for k, v in v_common.items()])
    
    tsne_plot_words('{} test - tSNE visualization of the {} vectors for A Study in Scarlet'.format(test.capitalize(), re.sub('_', ' ', version)), [k for k in v_proper.keys()] + [k for k in v_common.keys()], embeddings_en_2d, colors_dict, os.path.join(tSNE_output_folder, test, '{}_study_scarlet.png'.format(version)))

# RSA analysis

logging.info('Now carrying out the RSA analysis...')

all_correlations = defaultdict(dict)
rsa_plottable = defaultdict(dict)
tests = ['doppelganger_test', 'quality_test']
rsa_output_folder = 'results/RSA'
os.makedirs(rsa_output_folder, exist_ok=True)

for model in tqdm(folders):

    model_name = re.sub('_training/', '', model)
    model_folder = os.path.join(main_analyses_folder, model)
    rsa_model_results = defaultdict(dict)

    for t in tests:

        correlations = defaultdict(list)        

        for novel in os.listdir(model_folder):

            current_folder = os.path.join(model_folder, novel)

            for training_type in os.listdir(current_folder):

                if 'matched' in training_type and '.txt' not in training_type and training_type != 'common_nouns_matched':

                    training_folder = os.path.join(current_folder, training_type)

                    for root, directory, files in os.walk(training_folder):

                        for f in files:
                            #if '' in f and '.pickle' in f:
                            #elif 'wiki_' not in f and '.pickle' in f:
                            if t in root and '.pickle' in f and 'doppel' in t:
                                a_vectors = {k : v for k, v in (pickle.load(open('{}'.format(os.path.join(root, f)), 'rb'))).items() if k[len(k)-2:] == '_a'}
                                b_vectors = {k : v for k, v in (pickle.load(open('{}'.format(os.path.join(root, f)), 'rb'))).items() if k[len(k)-2:] == '_b'}
                            elif t in root and '.pickle' in f and 'qual' in t:
                                wiki_path = '{}'.format(os.path.join(root, f))
                                a_path = re.sub('quality', 'doppelganger', wiki_path)
                                a_vectors = {k : v for k, v in (pickle.load(open(a_path, 'rb'))).items() if k[len(k)-2:] == '_a'}
                                b_vectors = {re.sub('wiki', 'b', k) : v for k, v in (pickle.load(open('{}'.format(os.path.join(root, f)), 'rb'))).items() if k[len(k)-1] == 'i'}
                    a_vectors_ready = {k : v for k, v in a_vectors.items() if re.sub('_a$', '_b', k) in b_vectors.keys()}
                    b_vectors_ready = {k : v for k, v in b_vectors.items() if re.sub('_b$', '_a', k) in a_vectors.keys()}
                    if 'count' in model:
                        a_copy = a_vectors_ready.copy()
                        a_vectors_ready = {k: v.reshape(v.shape[1]) for k, v in a_copy.items()}
                        b_copy = b_vectors_ready.copy()
                        b_vectors_ready = {k: v.reshape(v.shape[1]) for k, v in b_copy.items()}
                    #if t == 'quality_test':
                        #import pdb; pdb.set_trace()

                    assert len(b_vectors_ready) == len(a_vectors_ready)

                    cosines_b = calculate_pairwise_comparisons(b_vectors_ready)
                    cosines_a = calculate_pairwise_comparisons(a_vectors_ready)

                    if len(cosines_b) > 2:
                        correlation = scipy.stats.spearmanr(cosines_b, cosines_a)
                    else:
                        correlation = [numpy.nan]
                    #wilcoxon_results = wilcoxon(cosines_wiki, cosines_a)
                    #print('Training type: {} - correlation: {}'.format(training_type, correlation[0]))
                    #correlations.append(correlation[0])
                    correlations[training_type].append(correlation[0])
        all_correlations[t] = correlations

    for test_name, correlations in all_correlations.items():
        rsa_test_results = defaultdict(float)
        test_short_name = re.sub('_test', '', test_name)
        test_output_folder = os.path.join(rsa_output_folder, test_short_name)
        os.makedirs(test_output_folder, exist_ok=True)
        #with open(os.path.join(test_output_folder, 'RSA.txt'), 'a') as o:
            #o.write('{} test\n\n'.format(test_short_name.capitalize()))

        for training_type, training_correlations in correlations.items():
            if training_type != 'common_nouns_matched':
                category = re.sub('_', ' ', re.sub('_matched|_unmatched', '', training_type)).capitalize()
                correlation_median = numpy.nanmedian(training_correlations)
                correlation_std = numpy.nanstd(training_correlations)
                #with open(os.path.join(test_output_folder, 'RSA_results.txt'.format(test_short_name)), 'a') as o:
                    #o.write('Model: {}\t-\t{}\n\nAverage Spearman correlation: {}\nMedian Spearman correlation: {}\nStandard deviation of the RSA results: {}\n\n'.format(re.sub('_', ' ', model_name).capitalize(), category, numpy.nanmean(training_correlations), correlation_median, numpy.nanstd(training_correlations)))

                rsa_test_results[category] = (correlation_median, correlation_std)
        rsa_model_results[test_short_name] = rsa_test_results
    rsa_plottable[model_name] = rsa_model_results

for test in tests:
    test_short_name = re.sub('_test', '', test)
    test_output_folder = os.path.join(rsa_output_folder, test_short_name)
    y_data = [(rsa_plottable[model_name][test_short_name]['Common nouns'][0], rsa_plottable[model_name][test_short_name]['Proper names'][0]) for model_name in folders]
    y_err = [(rsa_plottable[model_name][test_short_name]['Common nouns'][1], rsa_plottable[model_name][test_short_name]['Proper names'][1]) for model_name in folders]

    model_names = [re.sub('_training', '', model) for model in folders]
    if test_short_name == 'doppelganger':
        text_coords = (0, -15)
    else:
        text_coords = (0, 10)
    RSA_results_plots = MyBloodyPlots(output_folder=test_output_folder, font_folder=current_font_folder, x_variables=[re.sub('_', ' ', model_name).capitalize() for model_name in model_names], y_variables=y_data, x_axis='', y_axis='Median Spearman correlation', labels=['Common nouns', 'Proper names'], title='Median RSA correlations across spaces for the {} test'.format(test_short_name.capitalize()), identifier='RSA_results', colors=[teal, golden], y_lim=(0.0,1.0), x_ticks=True, y_ticks=True, y_grid=True, text_coords=text_coords)
    RSA_results_plots.plot_dat('two_lines')

    #o.write('\nTotal number of evaluations: {}\n\n'.format(len(training_correlations)))
