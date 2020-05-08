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

from data.mybloodyplots.mybloodyplots import MyBloodyPlots
from tqdm import tqdm
from matplotlib import pyplot
from matplotlib import font_manager as font_manager
from collections import defaultdict
from scipy import stats
from scipy.stats.morestats import wilcoxon
#from morestats import wilcoxon
from re import sub
#from nonce2vec.utils.novels_utilities import *
#from nonce2vec.utils.count_based_models_utils import cosine_similarity, normalise
from numpy import dot,sqrt,sum,linalg
from math import sqrt
from torch import Tensor
from collections import defaultdict

from sklearn.preprocessing import normalize as normalise

from numpy.linalg import norm

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
        logging.info('Not enough evaluations for {}!\nCheck the number of characters or common nouns for the current novel'.format(folder))

    return ranks


parser=argparse.ArgumentParser()
parser.add_argument('--pickles_folder', required=True, type = str, help = 'Specify the name of the folder where the pickles are contained')
parser.add_argument('--count_model_folder', required=True, type = str, help = 'Specify the name of the folder where the count vectors are stored')
parser.add_argument('--make_plots', required=False, action='store_true', help = 'Indicates whether to plot the results or not')
parser.add_argument('--write_to_file', required=False, action='store_true', help = 'Indicates whether to write to file or not')
args = parser.parse_args()

# Petty stuff setup

numpy.seterr(all='raise')
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Setting up the evaluation

cwd = os.getcwd()
folders = os.listdir(args.pickles_folder)

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
    big_folder = os.path.join(args.pickles_folder, folder)
    all_results = defaultdict(dict)
    logging.info('Current model: {}'.format(model))

    # Utilities
    doppelganger_entities_limit = defaultdict(int)
    quality_entities_limit = defaultdict(int)
    if 'count' in folder:
        with open(os.path.join(args.count_model_folder, 'count_wiki_2/count_wiki_2_cooccurrences.pickle'), 'rb') as word_cooccurrences_file:
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

            if args.write_to_file:
                output_file = '{}/{}/{}/{}/final_results.txt'.format(folder, novel, training, test)
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

        if args.write_to_file:
            across_path = 'results_per_model/{}/{}'.format(folder, test)
            number_of_novels_used = len(across_novels_results.keys())
            os.makedirs(across_path, exist_ok = True)
            with open('{}/details_per_training.txt'.format(across_path), 'a') as o:
                o.write('\nTest: {}\nLinguistic category considered: {}\n\nAcross-novels median of the medians: {}\nAcross-novels average of the averages: {}\nAcross-novels average of the MRRs: {}\nAcross-novels average standard deviation of the scores: {}\nAcross-novels median of the amount of entities considered: {}\nNumber of novels used: {}\n\nCorrelation with number of characters: {}\nCorrelation with the length of the novels: {}\nCorrelation with the standard deviation of the character\'s frequencies: {}\n'.format(re.sub('_', ' ', test), re.sub('_', ' ', training), across_median, across_average, across_average_mrr, across_average_std, across_entities, number_of_novels_used, number_of_characters_correlation, novel_length_correlation, std_of_characters_correlation))

        # Printing the results

        logging.info('\nTest: {}\nLinguistic category considered: {}\n\nAcross-novels median of the medians: {}\nAcross-novels average of the averages: {}\nAcross-novels average of the MRRs: {}\nAcross-novels average standard deviation of the scores: {}\nAcross-novels median of the amount of entities considered: {}\nNumber of novels used: {}\n\nCorrelation with number of characters: {}\nCorrelation with the length of the novels: {}\nCorrelation with the standard deviation of the character\'s frequencies: {}\n'.format(re.sub('_', ' ', test), re.sub('_', ' ', training), across_median, across_average, across_average_mrr, across_average_std, across_entities, number_of_novels_used, number_of_characters_correlation, novel_length_correlation, std_of_characters_correlation))

    final_results[model] = model_results
    histogram_results[model] = model_histogram
   
    '''

    # Comparing the statistical significance across different conditions

    significance_results = []
    try:
        significance_results.append([lsts[0][0], lsts[1][0], wilcoxon_results(list_one, list_two), numpy.median(list_one), numpy.median(list_two)])
    except ValueError:
        significance_results.append([lsts[0][0], lsts[1][0], ('Na', 'Na'), numpy.median(list_one), numpy.median(list_two)])
    try:
        significance_results.append([lsts[1][0], lsts[2][0], wilcoxon_results(list_two, list_three), numpy.median(list_two), numpy.median(list_three)])
    except ValueError:
        significance_results.append([lsts[1][0], lsts[2][0], ('Na', 'Na'), numpy.median(list_two), numpy.median(list_three)])
    try:
        significance_results.append([lsts[0][0], lsts[2][0], wilcoxon_results(list_one, list_three), numpy.median(list_one), numpy.median(list_three)])
    except ValueError:
        significance_results.append([lsts[0][0], lsts[2][0], ('Na', 'Na'), numpy.median(list_one), numpy.median(list_three)])
    #with open('{}/significance_test_results.txt'.format(path), 'a') as o:
    with open('{}/significance_test_results.txt'.format(path), 'a') as o:
        for s in significance_results:
            o.write('Comparison between:\n\n\t{} - median: {}\n\t{} - median: {}\n\nP-value: {}\nEffect size: {}\n\n\n'.format(s[0], s[3], s[1], s[4], s[2][0], s[2][1]))
    '''

# Reorganizing the data for plotting out the final results

# A reminder: model_results[setup_key] = {'average_rank' : across_average, 'median_rank' : across_median, 'average_mrr' : across_average_mrr, 'median_mrr' : across_median_mrr, 'corr_num_characters' : number_of_characters_correlation, 'corr_novel_length' : novel_length_correlation, 'corr_std_characters_mentions' : std_of_characters_correlation}
plottable_results = defaultdict(list)
novel_length = []
number_of_characters = []
std_of_character_mentions = []
golden = mcd.CSS4_COLORS['goldenrod']
teal = mcd.CSS4_COLORS['steelblue']
current_font_folder = '/import/cogsci/andrea/fonts'
models = [(re.sub('_', ' ', m)).capitalize() for m in final_results.keys()]
os.makedirs('plots', exist_ok=True)

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

main_folder = 'plots/main_results'
corr_folder = 'plots/correlations'
hist_folder = 'plots/histograms'
os.makedirs(main_folder, exist_ok=True)
os.makedirs(corr_folder, exist_ok=True)
os.makedirs(hist_folder, exist_ok=True)

# Plotting the main results

for test, results in plottable_results.items():
    results_plots = MyBloodyPlots(output_folder=main_folder, font_folder=current_font_folder, x_variables=models, y_variables=results, x_axis='', y_axis='Median rank', labels=['Common nouns', 'Proper names'], title='Median ranking results for the {} test'.format(test.capitalize()), identifier=test, colors=[teal, golden], y_invert=True, x_ticks=True)
    results_plots.plot_dat('two_lines')

# Plotting the correlational analyses

    corr_plot = MyBloodyPlots(output_folder=corr_folder, font_folder=current_font_folder, x_variables=models, y_variables=[novel_length, number_of_characters, std_of_character_mentions], x_axis='', y_axis='Spearman correlation', labels=['Novel length', 'Number of characters', 'Std of character mentions'], title='Correlational analysis for the results on proper names'.format(test.capitalize()), identifier='correlations', colors=['darkorange', 'orchid', 'darkgrey'], x_ticks=True)
    corr_plot.plot_dat('three_bars')

# Plotting the histogram analyses
    
for test_name, results in plottable_histogram.items():
    for results_index, variables_tuple in enumerate(results):
        model = models[results_index]
        hist_plots = MyBloodyPlots(output_folder=hist_folder, font_folder=current_font_folder, x_variables=[], y_variables=variables_tuple, x_axis='Median rank', y_axis='Frequency (N=59)', labels=['Common nouns', 'Proper names'], title='{} model - Histogram of the median ranks for the {} test'.format(model, test_name.capitalize()), identifier='{}_{}'.format(test_name.lower(), model.lower()), colors=[teal, golden], y_invert=False, y_ticks=True, x_ticks=True)
        hist_plots.plot_dat('histogram_two_sets')


import pdb; pdb.set_trace()

'''        
        plot_median[novel_name]=[]
        plot_mrr[novel_name]=[]
        lengths[novel_name]=[]
        names[novel_name]=[]
        characters_dict[novel_name]=[]
        characters_std[novel_name]=[]

        ambiguities={}
        ambiguities_present=False
        marker=False
        
        sentences_counter=[]
        ambiguities_counter=[]
        characters_frequency=[]
        base_novel = novel
        base_folder=os.listdir('{}/{}'.format(big, base_novel))
        novel_folder=os.listdir('{}/{}'.format(big, novel))
        for single_file_or_folder in novel_folder:
            if '{}'.format(test) in single_file_or_folder:
                test_folder = os.listdir('{}/{}/{}'.format(big, novel, single_file_or_folder))
            #if 'evaluation' in single_file and '{}'.format(test) in single_file:
            #if 'evaluation' in single_file and 'quality' not in single_file:
                for f in test_folder:
                    if 'evaluation' in f:
                        evaluation=open('{}/{}/{}/{}'.format(big, novel, single_file_or_folder, f)).readlines()
                        if 'common' in big:
                            equivalent_folder = re.sub('common_matched_|common_unmatched', '', big)
                            proper_file = open('{}/{}/{}/{}'.format(equivalent_folder, novel, single_file_or_folder, f)).readlines()
                            proper_list = [l.strip() for l in proper_file]
                            relevant_line = proper_list[3]
                            relevant_number = int(relevant_line.split('\t')[1])
        
                        marker=False
                        if len(evaluation)>1:
                            marker=True
                        if marker==True:
                            line1=evaluation[0].strip('\n').split('\t')[1]
                            line2=evaluation[1].strip('\n').split('\t')[1]
                            line3=evaluation[2].strip('\n').split('\t')[1]
                            line4=evaluation[3].strip('\n').split('\t')[1]
                            #mrr+=float(line1)
                            #median+=float(line2)
                            list_var_mrr.append(float(line1))
                            list_var_median.append(float(line2))
                            list_var_mean.append(float(line3))
                            plot_median[novel_name] = float(line2)
                            plot_mrr[novel_name].append(float(line1))
                            #characters+=int(line3)
                            characters.append(int(line4))
            
        if marker==True:
            for single_file in base_folder:
                if 'character' in single_file and 'ender' not in single_file:
                    characters_file=open('{}/{}/{}'.format(big, base_novel, single_file)).readlines()
                    for l in characters_file:
                        l=l.split('\t') 
                        l=int(l[0])
                        if l>=10:
                            characters_frequency.append(l)
                if 'data_output' in single_file:
                    data_output_filenames=os.listdir('{}/{}/data_output'.format(big, base_novel))
                    if 'ambiguities' in data_output_filenames:
                        ambiguities_present=True
                        ambiguities_filenames=os.listdir('{}/{}/data_output/ambiguities'.format(big, base_novel))
                        for ambiguity in ambiguities_filenames:
                            current_ambiguity=open('{}/{}/{}/data_output/ambiguities/{}'.format(big, base_novel, ambiguity)).readlines()
                            for character_line in current_ambiguity:
                                if 'too: ' in character_line:
                                    character_line=character_line.strip('\n').split('too: ')[1]
                                    character_ambiguity=character_line.split(' out of ')[0]
                                    sent=character_line.split(' out of ')[1].strip('\n').replace(' sentences', '')
                                    ambiguities_counter.append(int(character_ambiguity))
                                    sentences_counter.append(int(sent))
                if numpy.sum(sentences_counter)==0:
                    ambiguities_present=False
            if ambiguities_present==True:
                novel_ambiguity=numpy.sum(ambiguities_counter)
                total_sentences=numpy.sum(sentences_counter)
                percentage=round((float(novel_ambiguity)*100.0)/float(total_sentences), 2)
                ambiguities[novel_name]=[novel_ambiguity, total_sentences, percentage]   

            original_file = [k for k in  os.listdir('{}/{}/novel'.format(big, base_novel)) if 'replication' in k][0] 
            open_file=open('{}/{}/novel/{}'.format(big, base_novel, original_file)).read()
            open_file=sub(r'\W+', ' ', open_file)
            open_file=open_file.split(' ')
            novel_length=len(open_file)
            if type(novel_length) != list:
                lengths[novel_name]=novel_length
            else:
                print(novel_name)
            names[novel_name].append(novel)
            characters_dict[novel_name] = int(line4)
            #print(novel_name)
            #print(len(characters_frequency))
            std_characters_frequency=numpy.std(characters_frequency)
            characters_std[novel_name].append(std_characters_frequency)
        if marker==False:
            print(novel_name)
            pass
        #import pdb; pdb.set_trace()
    average_mrr=numpy.median(list_var_mrr)
    var_mrr=numpy.var(list_var_mrr)
    average_median=numpy.median(list_var_median)
    var_median=numpy.var(list_var_median)
    average_mean=numpy.median(list_var_mean)
    var_mean=numpy.var(list_var_mean)
    average_characters=int(round(numpy.mean(characters)))
    with open('pickles_for_significance_testing/{}/{}_test_{}.pickle'.format(test, test, args.training_mode), 'wb') as o:
        pickle.dump(list_var_median, o)

    if 'common' not in args.training_mode:
        if average_characters>1.0:
            setup_medians = [v for k, v in plot_median.items() if v != []]
            #print(setup_medians)
            setup_lengths = [v for k,v in lengths.items() if v != []]
            #print(setup_lengths)
            setup_chars = [v for k,v in characters_dict.items() if v != []]
            #print(setup_chars)
            setup_std = [v for k,v in characters_std.items() if v != []]
            #print(setup_std)
            spearman_lengths=round(scipy.stats.spearmanr(setup_lengths, setup_medians)[0],2) 
            spearman_chars=round(scipy.stats.spearmanr(setup_chars, setup_medians)[0],2)
            spearman_std = round(scipy.stats.spearmanr(setup_std, setup_medians)[0],2)
            print('\nSetup: {}\n\nMedian MRR: {}\nMRR Variance: {}\nMedian Median: {}\nVariance in median median: {}\nMedian of means: {}\nMedian of means variance: {}\nAverage number of characters: {}\nTotal of rankings taken into account: {}\nCorrelation with length: {}\nCorrelation with number of characters: {}\nCorrelation with standard deviation of characters frequency: {}\n'.format(setup_shorthand, average_mrr, var_mrr, average_median, var_median, average_mean, var_mean, average_characters, len(list_var_mrr), spearman_lengths, spearman_chars, spearman_std))

    else:
        if average_characters>1.0:
            setup_medians = [v for k, v in plot_median.items() if v != []]
            #print(setup_medians)
            print('\nSetup: {}\n\nMedian MRR: {}\nMRR Variance: {}\nMedian Median: {}\nVariance in median median: {}\nMedian of means: {}\nMedian of means variance: {}\nAverage number of characters: {}\nTotal of rankings taken into account: {}\n'.format(setup_shorthand, average_mrr, var_mrr, average_median, var_median, average_mean, var_mean, average_characters, len(list_var_mrr)))

        if args.make_plots:
            results_output=open('{}/doppel_results_{}.txt'.format(output_folder, novel),'w')
            results_output.write('\nSetup: {}\n\nMedian MRR: {}\nMRR Variance: {}\nMedian Median: {}\nVariance in median median: {}\nMedian of means: {}\nMedian of means variance: {}\nAverage number of characters: {}\nTotal of rankings taken into account: {}\n'.format(setup_shorthand, average_mrr, var_mrr, average_median, var_median, average_mean, var_mean, average_characters, len(list_var_mrr)))
        if ambiguities_present==True:
            if len(ambiguities.keys())>0 and total_evaluations_runs_counter==0:
                ambiguity_percentages=[]
                for amb in ambiguities.keys():
                    novel_amb=ambiguities[amb]
                    amb_sent=novel_amb[0]
                    total_sent=novel_amb[1]
                    percent=novel_amb[2]
                    ambiguity_percentages.append(percent)
                final_percent=numpy.mean(ambiguity_percentages)
                print('Percentage of ambiguous sentences of all sentences used for training (containing more than one character): {} %\n'.format(round(final_percent, 3)))

                total_evaluations_runs_counter+=1
'''
if args.make_plots:

    if 'bert' in args.training_mode:
        output_folder='{}_layer_{}_doppelgaenger_test_plots'.format(args.training_mode, args.bert_layer)
    else:
        output_folder='{}_{}_test_plots'.format(args.training_mode, test)
    os.makedirs(output_folder, exist_ok=True)

    ### Ambiguity infos


    with open('{}/doppel_ambiguities_info.txt'.format(output_folder), 'w') as ambiguity_file:
        if len(ambiguities.keys())>0:
            for amb in ambiguities.keys():
                novel_amb=ambiguities[amb]
                amb_sent=novel_amb[0]
                total_sent=novel_amb[1]
                perc_amb=novel_amb[2]
                ambiguity_file.write('\nAmbiguous sentences: {} out of {}\nPercentage: {} %\n\n'.format(amb_sent, total_sent, perc_amb))

    def ticks(setups_dict, mode):
        max_value=[]
        for i in setups_dict.keys():
            max_i=max(setups_dict[i])
            max_value.append(max_i) 
        x_ticks=[u for u in range(1, len(setups_dict[i])+1)]
        if mode=='median':
            y_ticks=[i for i in range(1, int(max(max_value))+1, 2)] 
        elif mode=='mrr':
            y_ticks=[i for i in numpy.linspace(0, 1, 11)] 
        return x_ticks, y_ticks

    short_names=[]
    for n in names[setup]:
        n=n.replace('_', ' ').split(' by')
        short_names.append(n[0])

    sum_spearman='N.A.'

    ### Novels data:

    novels_info=open('{}/{}_novels_info.txt'.format(output_folder, setup_shorthand), 'w')
    for n in [k for k in range(0, len(short_names))]:
        novels_info.write('Name:\t{}\nLength in words:\t{}\nCharacters evaluated:\t{}\nMedian Rank:\t{}\nMRR:\t{}\nStandard deviation of characters frequency: {}\n\n'.format(short_names[n],lengths[setup][n],characters_dict[setup][n],list_var_median[n],list_var_mrr[n], characters_std[setup][n]))

    sum_color=['lightslategrey', 'coral', 'darkgoldenrod', 'darkcyan', 'deeppink', 'lightgreen', 'aquamarine', 'green', 'purple', 'gold', 'sienna', 'olivedrab']

    ### Lenghts/score

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup],lengths[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], lengths[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup],lengths[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2V: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], lengths[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_median, 'median')

    plt.xlabel('Median Rank')
    plt.ylabel('Novel length (words)')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    #plt.yticks([(((i+999)/1000)*1000) for i in numpy.linspace(0,max(lengths[setup]),10)])
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_lenghts.png'.format(output_folder), dpi=1200, format='png', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Number of characters/score

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup], characters_dict[setup], color=sum_color, label=legend_label, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_dict[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup], characters_dict[setup], color=sum_color, label=legend_label, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_dict[setup])[0],2)) 
    x_ticks, y_ticks=ticks(plot_median, 'median')

    plt.xlabel('Median Rank')
    plt.ylabel('Number of characters')
    #plt.yticks(characters_dict[setup] )
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_characters.png'.format(output_folder), dpi=1200, format='png', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Variance of characters frequency/score

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup], characters_std[setup], label=legend_label, color=sum_color, marker= 'P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_std[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup], characters_std[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_median[setup], characters_std[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_median, 'median')

    plt.xlabel('Median Rank')
    plt.ylabel('Variance of character mention frequency')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    #plt.yticks(characters_std[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_frequency_std.png'.format(output_folder), dpi=1200, format='png', pad_inches=0.2, bbox_inches='tight')

    plt.clf()

    ### Score/novel

    for setup in plot_median.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_median[setup], short_names, label=legend_label, color=sum_color, marker='P')
        else:
            legend_label='N2V'
            plt.scatter(plot_median[setup], short_names, label=legend_label, color=sum_color, marker='v')
    plt.xlabel('Median Rank')
    plt.ylabel('Novel')
    #plt.xticks(y_ticks)
    #plt.yticks(short_names )
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_median_names.png'.format(output_folder, setup, novel), dpi=1200, format='png', bbox_inches='tight', pad_inches=0.2)
    plt.clf()

    ### Lenghts/score MRR

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter( plot_mrr[setup],lengths[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup], lengths[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter( plot_mrr[setup],lengths[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup], lengths[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
    plt.xlabel('MRR')
    plt.ylabel('Novel length (words)')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    plt.gca().invert_xaxis()
    #plt.yticks(lengths[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_mrr_lengths.png'.format(output_folder), dpi=1200, format='png', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Number of characters/score MRR

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter( plot_mrr[setup], characters_dict[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup],characters_dict[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter( plot_mrr[setup], characters_dict[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup],characters_dict[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
    plt.xlabel('MRR')
    plt.ylabel('Number of characters')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    plt.gca().invert_xaxis()
    #plt.yticks(characters_dict[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_MRR_characters.png'.format(output_folder), dpi=1200, format='png', bbox_inches='tight', pad_inches=0.2)

    plt.clf()


    ### Variance of characters frequency/score MRR

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter( plot_mrr[setup], characters_std[setup], label=legend_label, color=sum_color, marker='P') 
            sum_spearman='Sum: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup], characters_std[setup])[0],2)) 
        else:
            legend_label='N2V'
            plt.scatter( plot_mrr[setup], characters_std[setup], label=legend_label, color=sum_color, marker='v') 
            n2v_spearman='N2v: {}'.format(round(scipy.stats.spearmanr(plot_mrr[setup],characters_std[setup])[0],2)) 
    #x_ticks, y_ticks=ticks(plot_mrr, 'mrr')
    plt.xlabel('MRR')
    plt.ylabel('Variance of character mention frequency')
    plt.title('Spearman correlations: {} - {}'.format(sum_spearman, n2v_spearman))
    plt.gca().invert_xaxis()
    #plt.yticks(characters_std[setup] )
    #plt.xticks(y_ticks)
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_MRR_characters_std.png'.format(output_folder), dpi=1200, format='png', bbox_inches='tight', pad_inches=0.2)

    plt.clf()

    ### Score mrr /novel

    for setup in plot_mrr.keys():
        if 'sum' in setup:
            legend_label='Sum'
            plt.scatter(plot_mrr[setup], short_names, label=legend_label, color=sum_color, marker='P')
        else:
            legend_label='N2V'
            plt.scatter(plot_mrr[setup], short_names, label=legend_label, color=sum_color, marker='v')
    plt.xlabel('MRR')
    plt.ylabel('Novel')
    plt.gca().invert_xaxis()
    #plt.xticks(y_ticks)
    #plt.yticks(short_names )
    plt.legend(loc='best', ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('{}/doppel_MRR_names.png'.format(output_folder), dpi=1200, format='png', bbox_inches='tight', pad_inches=0.2)
    plt.clf()
