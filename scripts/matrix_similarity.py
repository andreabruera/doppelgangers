import numpy
import pickle
import argparse
import os
import scipy
import nonce2vec
import tqdm
import re
import collections

from tqdm import tqdm
from scipy.stats import spearmanr
from scipy.stats.morestats import wilcoxon
from nonce2vec.utils.count_based_models_utils import cosine_similarity, normalise
from collections import defaultdict

def calculate_pairwise_comparisons(vectors):
    cosines = []
    for first_name, first_vector in vectors.items():
        for second_name, second_vector in vectors.items():
            if first_name != second_name:
                cosines.append(cosine_similarity(normalise(first_vector), normalise(second_vector)))
    return cosines

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type = str, required = True, help = 'Location where to find the vectors pickle')
args = parser.parse_args()

folders = os.listdir(args.folder)
#model_name = re.sub('_novels/', '', args.folder)
model_name = re.sub('_training/', '', args.folder)
#correlations = []
all_correlations = defaultdict(dict)
tests = ['doppelganger_test', 'quality_test']


for t in tests:
    correlations = defaultdict(list)

    for novel in tqdm(folders):

        current_folder = '{}/{}'.format(args.folder, novel)

        for training_type in os.listdir(current_folder):

            if 'matched' in training_type and '.txt' not in training_type:

                training_folder = '{}/{}'.format(current_folder, training_type)

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
                #if t == 'quality_test':
                    #import pdb; pdb.set_trace()

                assert len(b_vectors_ready) == len(a_vectors_ready)

                cosines_b = calculate_pairwise_comparisons(b_vectors_ready)
                cosines_a = calculate_pairwise_comparisons(a_vectors_ready)

                correlation = spearmanr(cosines_b, cosines_a)
                #wilcoxon_results = wilcoxon(cosines_wiki, cosines_a)
                print('Training type: {} - correlation: {}'.format(training_type, correlation[0]))
                #correlations.append(correlation[0])
                correlations[training_type].append(correlation[0])
    all_correlations[t] = correlations


for test_name, correlations in all_correlations.items():
    for training_type, training_correlations in correlations.items():
        with open('results_per_model/{}/{}/RSA.txt'.format(model_name, test_name), 'a') as o:
            #o.write('Setup: {}\n\nSpearman correlation: {}\nWilcoxon: {}'.format(args.folder, correlation_results, wilcoxon_results))
            o.write('Setup: {}\t-\t{}\n\nAverage Spearman correlation: {}\nMedian Spearman correlation: {}\nStandard deviation of the RSA results: {}\n'.format(args.folder, training_type, numpy.nanmean(training_correlations), numpy.nanmedian(training_correlations), numpy.nanstd(training_correlations)))

            o.write('\nTotal number of evaluations: {}\n\n'.format(len(training_correlations)))
