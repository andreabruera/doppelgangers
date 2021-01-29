import collections
import re
import numpy
import os
import pickle
import random
import scipy
import tqdm
import multiprocessing
import logging

from tqdm import tqdm
from utils import get_novels_paths, get_word_vectors, get_scores, load_model, \
                  process_novel, sentence_permutation

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)
os.makedirs('temp', exist_ok=True)
chosen_metric = 'pearson_r'
#chosen_metric = 'cosine_similarity'
categories = ['common nouns', 'characters']

for model_name in ['bert', 'gpt2', 'w2v', 'elmo']:
#for model_name in ['w2v']:

    logging.info('Testing on {}, with distance metric {}'.format(model_name, chosen_metric))
    logging.info('Now loading the model')
    model = load_model(model_name)

    logging.info('Now extracting the paths')
    ### Extracting paths from the novel aficionados dataset folder
    paths = [[tup[0], tup[1], model_name, model, chosen_metric] for tup in get_novels_paths('/import/cogsci/andrea/github/novel_aficionados_dataset').items()]

    ### Starting the experiment
    logging.info('Now starting the experiment')

    final_results = collections.defaultdict(dict)
    final_stats = collections.defaultdict(dict)

    for path in tqdm(paths):

        novel_name = path[0]

        novel_sentences, max_entity_length, max_sentences_length = process_novel(path)

        novel_scores = collections.defaultdict(dict)
        novel_stats = collections.defaultdict(dict)

        for category, vectors_per_sentence in novel_sentences.items():

            category_scores = collections.defaultdict(list)
            category_stats = collections.defaultdict(list)

            arguments = [(vectors_per_sentence, max_sentences_length, chosen_metric) for i in range(300)]

            if __name__ == '__main__':

                pool = multiprocessing.Pool()
                results_collector = pool.imap(func=sentence_permutation, iterable=arguments)
                pool.close()
                pool.join()

            for result in results_collector:

                if len(result) == 4:

                    category_scores['pure ranking'].append(result[1])
                    category_scores['normalized ranking'].append(result[2])
                    category_scores['rsa score'].append(result[3])

                    category_stats['standard deviation of number of per-entity sentences'] = result[0]
         
            category_stats['number of sentences'] = max_sentences_length
            category_stats['number of entities'] = max_entity_length

            novel_scores[category] = category_scores
            novel_stats[category] = category_stats

        final_results[novel_name] = novel_scores
        final_stats[novel_name] = novel_stats

    ### Preparing results for printing them out
    printable_results = collections.defaultdict(dict)

    for category in categories:

        category_results = collections.defaultdict(list)

        for novel_name, novel_dict in final_results.items():

            for score_type, scores in novel_dict[category].items():

                unpacked_scores = [numpy.nanmean(score) for score in scores] if score_type != 'rsa score' else scores
                scores_mean = numpy.nanmean(unpacked_scores)
                scores_median = numpy.nanmedian(unpacked_scores)
                scores_std = numpy.nanstd(unpacked_scores)
                category_results[score_type].append([scores_mean, scores_median, scores_std])

        '''
        category_stats = collections.defaultdict(dict)
        correlation_results = collections.defaultdict(lambda : collections.defaultdict(list))

                for value_type, value_results in statistics_collector[novel].items():
                    correlation_results[value_type][score_type].append([numpy.nanmean(printable_results[score_type], axis=0), value_results])
        '''
    ### Pickling results
    with open(os.path.join('temp', 'final_results_and_stats_{}_{}.pkl'.format(model_name, chosen_metric)), 'wb') as o:
        pickle.dump([final_results, final_stats], o)

    print('Now writing to file...')
    ### Writing results to file
    with open(os.path.join('temp', 'final_results_{}_{}.txt'.format(model_name, chosen_metric)), 'w') as o:
        o.write('Model: {}\n\n\n'.format(model_name))

        for category, score_type_dict in printable_results.items():
            ### Writing main results to file
            o.write('Category: {}\n\n'.format(category))
            for score_type, scores in score_type_dict.items():
                o.write('{}\n'
                '\tMean score: {}\n'
                '\tMedian score: {}\n'
                '\tStandard deviation: {}\n\n'
                ''.format(score_type.capitalize(), scores[0], scores[1], scores[2]))

                '''
                ### Writing correlation results to file
                for value_type, all_results in correlation_results.items():
                    value_results = all_results[score_type]
                    if score_type == 'pure ranking':
                        current_r = scipy.stats.pearsonr([k[0] for k in value_results], [k[1] for k in value_results])[0]
                        current_rho = scipy.stats.spearmanr([k[0] for k in value_results], [k[1] for k in value_results])[0]
                    ### Inverting the scores for rsa and normalized values because they go in the opposite directions to the ther variables
                    else:
                        other_scores = [k[1] for k in value_results]
                        maximum = max(other_scores)
                        normalized_ranking = [(k - 1) / (maximum - 1) for k in other_scores]
                        current_r = scipy.stats.pearsonr([k[0] for k in value_results], normalized_ranking)[0]
                        current_rho = scipy.stats.spearmanr([k[0] for k in value_results], normalized_ranking)[0]
                    o.write('\tCorrelation with {}: pearson {}, spearman {}\n'.format(value_type, current_r, current_rho))
                o.write('\n')
                '''
            o.write('\n')

