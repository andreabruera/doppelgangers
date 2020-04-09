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
number_of_characters = defaultdict(dict)
doppelganger_entities_limit = defaultdict(int)
quality_entities_limit = defaultdict(int)

novel_lengths = defaultdict(int)
characters_std = defaultdict(float)

if 'count' in args.folder:
    with open('count_models/count_wiki_2/count_wiki_2_cooccurrences.pickle', 'rb') as word_cooccurrences_file:
        background_vectors_length = max(dill.load(word_cooccurrences_file).keys())
        logging.info(background_vectors_length)

for setup in setups:
    
    logging.info('Currently evaluating performance on setup: {}'.format(setup))

    training = setup[0]
    test = setup[1]
    setup_key = '{}_{}'.format(test, training)
    current_results = defaultdict(dict)
    '''
    plot_median = defaultdict(float)
    plot_mrr={}
    lengths= defaultdict(int)
    names={}
    characters_dict = defaultdict(int)
    characters_std={}
    total_evaluations_runs_counter=0
    characters=[]
    list_var_mrr=[]
    list_var_median=[]
    list_var_mean=[]
    '''

    for novel in os.listdir(big_folder):
        #print(novel)
        #setup_shorthand=big.replace(cwd, '') 
        novel_folder = '{}/{}'.format(big_folder, novel)
        novel_number = re.sub('_no_header.txt', '', [filename for filename in os.listdir('{}/novel'.format(novel_folder)) if 'no_header' in filename][0])

        test_folder = '{}/{}/{}'.format(novel_folder, training, test)
        try:
            current_pickle = pickle.load(open('{}/{}.pickle'.format(test_folder, novel_number), 'rb'))
            if 'count' in args.folder:
                entity_dict = defaultdict(numpy.ndarray)
                for e, v in current_pickle.items():
                    entity_dict[e] = v[0][:background_vectors_length]
            else:
                entity_dict = current_pickle

            if test == 'quality_test':
                full_novel_pickle = pickle.load(open('{}/{}/doppelganger_test/{}.pickle'.format(novel_folder, training, novel_number), 'rb'))
                for e, v in full_novel_pickle.items():
                    if e[-1] == 'a':
                        if 'count' in args.folder:
                            entity_dict[e] = v[0][:background_vectors_length]
                        else:
                            entity_dict[e] = v

            if test == 'doppelganger_test':
                if 'proper' not in training:
                    current_results[novel] = evaluation(entity_dict, test_folder, doppelganger_entities_limit[novel])
                else:
                    current_results[novel] = evaluation(entity_dict, test_folder)
                    doppelganger_entities_limit[novel] = int(len(current_results[novel])/2)
            else:
                if quality_entities_limit[novel] > 0:
                    current_results[novel] = evaluation(entity_dict, test_folder, quality_entities_limit[novel])
                    quality_entities_limit[novel] = min(quality_entities_limit[novel], len(current_results[novel])/2)
                else:
                    current_results[novel] = evaluation(entity_dict, test_folder)
                    quality_entities_limit[novel] = int(len(current_results[novel])/2)

                if training == 'proper_names_matched':
                    novel_lengths[novel] = len([w for l in open('{}/novel/{}_no_header.txt'.format(novel_folder, novel_number)).readlines() for w in l.split()])
                    characters_std[novel] = numpy.nanstd([int(l.split('\t')[0]) for l in open('{}/characters_list_{}.txt'.format(novel_folder, novel_number)).readlines() if int(l.split('\t')[0]) >= 10])
            assert len(current_results[novel]) % 2 == 0
        except FileNotFoundError:
            logging.info('Could not find the file for {}...'.format(novel))
    all_results[setup_key] = current_results


'''
doppelganger_entities_amount = defaultdict(int)
for n, lst in all_results['doppelganger_test_proper_names_matched'].items():
    doppelganger_entities_amount[n] = int(len(lst)/2)

quality_entities_amount = defaultdict(int)
proper_entities_amount = defaultdict(int)
common_entities_amount = defaultdict(int)
for n, lst in all_results['quality_test_proper_names_matched'].items():
    proper_entities_amount[n] = int(len(lst)/2)
for n, lst in all_results['quality_test_common_nouns_matched'].items():
    common_entities_amount[n] = int(len(lst)/2)
for n in proper_entities_amount.keys():
    quality_entities_amount[n] = min(proper_entities_amount[n], common_entities_amount[n])

filtered_results = defaultdict(dict)

for setup in setups:
    training = setup[0]
    test = setup[1]
    setup_key = '{}_{}'.format(test, training)
    if test == 'doppelganger_test' and 'common_nouns' in training:
        filtered_novels = defaultdict(list)
        for n, r in all_results[setup_key].items():
            part_a = [k for k in r if k[0][-1] == 'a'][:doppelganger_entities_amount[n]]
            part_b = [k for k in r if k[0][-1] == 'b'][:doppelganger_entities_amount[n]]
            for a, b in zip(part_a, part_b):
                filtered_novels[n].append(a)
                filtered_novels[n].append(b)
        filtered_results[setup_key] = filtered_novels
    elif test == 'doppelganger_test' and 'proper_names' in training:
        filtered_results[setup_key] = all_results[setup_key]
    else:
        filtered_novels = defaultdict(list)
        for n, r in all_results[setup_key].items():
            part_a = [k for k in r if k[0][-1] == 'a'][:quality_entities_amount[n]]
            part_b = [k for k in r if k[0][-1] == 'b'][:quality_entities_amount[n]]
            for a, b in zip(part_a, part_b):
                filtered_novels[n].append(a)
                filtered_novels[n].append(b)
        filtered_results[setup_key] = filtered_novels

#with open('filtered_results.pickle', 'wb') as o:
with open('filtered_results.pickle', 'rb') as o:
    #pickle.dump(filtered_results, o)
    filtered_results = pickle.load(o)
'''
filtered_results = all_results
post_aggregation_results_median = defaultdict(list)
overall_aggregation_results_median = defaultdict(list)

for setup in setups:
    training = setup[0]
    test = setup[1]
    setup_key = '{}_{}'.format(test, training)
    results = filtered_results[setup_key]
    across_novels_results = defaultdict(list)
    for novel, results_list in results.items():
        figures = [k[1] for k in results_list]
        reciprocal_ranks = [1/k[1] for k in results_list]
        amount_of_entities = len(results_list)/2
        novel_median = numpy.median(figures)
        novel_average = numpy.average(figures)
        novel_mrr = numpy.average(reciprocal_ranks)
        novel_std = numpy.std(figures)
        output_file = '{}/{}/{}/{}/final_results.txt'.format(args.folder, novel, training, test)
        with open(output_file, 'w') as o:
            o.write('Test: {}\nResults for: {}\nModel used: {}\n\nWithin-novel median: {}\nWithin-novel average: {}\nWithin-novel MRR: {}\nWithin-novel standard deviation of the scores: {}\nAmount of entities considered: {}'.format(re.sub('_', ' ', test), novel, re.sub('_', ' ', training), novel_median, novel_average, novel_mrr, novel_std, amount_of_entities))
        across_novels_results[novel] = [novel_median, novel_average, novel_mrr, novel_std, amount_of_entities]
    list_of_medians = [m[0] for n, m in across_novels_results.items()]
    across_median = numpy.nanmedian(list_of_medians)
    across_average = numpy.nanmean([m[1] for n, m in across_novels_results.items()])
    across_mrr = numpy.nanmean([m[2] for n, m in across_novels_results.items()])
    across_std = numpy.nanmedian([m[3] for n, m in across_novels_results.items()])
    list_of_entities = [m[4] for n, m in across_novels_results.items()]
    across_entities = numpy.nanmedian(list_of_entities)

    try:
        number_of_characters_correlation = scipy.stats.spearmanr(list_of_medians, list_of_entities)[0]
        novel_length_correlation = scipy.stats.spearmanr(list_of_medians, [k[1] for k in novel_lengths.items()])[0]
        std_of_characters_correlation = scipy.stats.spearmanr(list_of_medians, [k[1] for k in characters_std.items()])[0]
    except FloatingPointError:
        number_of_characters_correlation = 'nan'
        novel_length_correlation = 'nan'
        std_of_characters_correlation = 'nan'
    across_path = 'results_per_model/{}/{}'.format(args.folder, test)
    number_of_novels_used = len(across_novels_results.keys())
    os.makedirs(across_path, exist_ok = True)
    with open('{}/details_per_training.txt'.format(across_path), 'a') as o:
        o.write('\nTest: {}\nModel used: {}\n\nAcross-novels median of the medians: {}\nAcross-novels average of the averages: {}\nAcross-novels average of the MRRs: {}\nAcross-novels median standard deviation of the scores: {}\nAcross-novels median of the amount of entities considered: {}\nNumber of novels used: {}\n\nCorrelation with number of characters: {}\nCorrelation with the length of the novels: {}\nCorrelation with the standard deviation of the character\'s frequencies: {}\n'.format(re.sub('_', ' ', test), re.sub('_', ' ', training), across_median, across_average, across_mrr, across_std, across_entities, number_of_novels_used, number_of_characters_correlation, novel_length_correlation, std_of_characters_correlation))
    post_aggregation_median = [m[0] for n, m in across_novels_results.items()]
    overall_aggregation_median = [r[1] for n, lst in results.items() for r in lst]
    post_aggregation_results_median[test].append([training, post_aggregation_median])
    overall_aggregation_results_median[test].append([training, overall_aggregation_median])

to_be_plotted = defaultdict(dict)
to_be_plotted['novel_by_novel'] = post_aggregation_results_median
to_be_plotted['overall'] = overall_aggregation_results_median

for analysis_type, results in to_be_plotted.items():
    for t, lsts in results.items():
        path = 'results_per_model/{}/{}/{}'.format(args.folder, t, analysis_type)
        os.makedirs(path, exist_ok = True)

        length = int(min(len(lsts[0][1]), len(lsts[1][1]), len(lsts[2][1])))
        maximum = int(max(max(lsts[0][1]), max(lsts[1][1]), max(lsts[2][1])))
        if analysis_type == 'overall':
            quantile = int(max(numpy.quantile(lsts[0][1], 0.95), numpy.quantile(lsts[1][1], 0.95), numpy.quantile(lsts[2][1], 0.95)))
        else:
            quantile = int(max(numpy.quantile(lsts[0][1], 1), numpy.quantile(lsts[1][1], 1), numpy.quantile(lsts[2][1], 1)))
        list_one = lsts[0][1][:length]
        list_two = lsts[1][1][:length]
        list_three = lsts[2][1][:length]

        golden = mcd.CSS4_COLORS['goldenrod']
        teal = mcd.CSS4_COLORS['steelblue']

        barWidth = 0.2
        pyplot.clf()
        pyplot.xlabel('Median evaluation score')
        pyplot.ylabel('Frequency')
        position_one = [a for a in range(1,maximum+1)]
        #pyplot.hist([list_one,list_two,list_three], bins = numpy.arange(quantile)+1, range = (0, quantile), label = [re.sub('_', ' ', lsts[0][0]).capitalize(), re.sub('_', ' ', lsts[1][0]).capitalize(), re.sub('_', ' ', lsts[2][0]).capitalize()], align = 'mid', edgecolor = 'white')
        pyplot.hist([list_one, list_three], bins = numpy.arange(quantile)+1, range = (0, quantile), label = [re.sub('_', ' ', lsts[0][0]).capitalize(), re.sub('_', ' ', lsts[2][0]).capitalize()], align = 'mid', edgecolor = 'white', color = [golden, teal])
        
        #pyplot.title('Results for the test:    {}\nModel:    {}\nAnalysis type:    {}'.format(re.sub('_', ' ', t).capitalize(), re.sub('_training', '', args.folder).capitalize(), re.sub('_', ' ', analysis_type).capitalize()), wrap = True, multialignment = 'left')
        pyplot.title('Results for the {}\nModel:    {}\n'.format(re.sub('_', ' ', t).capitalize(), re.sub('_training', '', args.folder).capitalize()), wrap = True, multialignment = 'left')
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.subplots_adjust(top=0.85)
        pyplot.savefig('{}/histogram_plot.png'.format(path, ))

        significance_results = []
        significance_results.append([lsts[0][0], lsts[1][0], wilcoxon_results(list_one, list_two), numpy.median(list_one), numpy.median(list_two)])
        significance_results.append([lsts[1][0], lsts[2][0], wilcoxon_results(list_two, list_three), numpy.median(list_two), numpy.median(list_three)])
        significance_results.append([lsts[0][0], lsts[2][0], wilcoxon_results(list_one, list_three), numpy.median(list_one), numpy.median(list_three)])
        #with open('{}/significance_test_results.txt'.format(path), 'a') as o:
        with open('{}/significance_test_results.txt'.format(path), 'a') as o:
            for s in significance_results:
                o.write('Comparison between:\n\n\t{} - median: {}\n\t{} - median: {}\n\nP-value: {}\nEffect size: {}\n\n\n'.format(s[0], s[3], s[1], s[4], s[2][0], s[2][1]))
    

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
