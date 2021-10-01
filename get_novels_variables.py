import os
import pickle
import pandas
import numpy

from utils import get_novels_paths, get_word_vectors, get_scores, load_model, \
                  process_novel, sentence_permutation


paths = [[tup[0], tup[1],  '', '', ''] for tup in get_novels_paths('/import/cogsci/andrea/github/novel_aficionados_dataset').items()]

novel_scores = dict()
actual_novels = list()

for model in ['bert', 'elmo', 'w2v_cosine_similarity', 'gpt2_pearson_r']:
    if 'w2v' in model:
        m = pickle.load(open(os.path.join('temp', 'final_results_{}.pkl'.format(model)), 'rb'))[0]['characters']
    else:
        m = pickle.load(open(os.path.join('temp', 'final_results_{}.pkl'.format(model)), 'rb'))['characters']
    model_dict = dict()
    for path in paths:
        novel_name = path[0]
        if novel_name in m.keys():
            if 'w2v' in model:
                score = numpy.average([numpy.average(iteration) for iteration in m[novel_name]['normalized ranking'][1]])
            else:
                score = numpy.average(m[novel_name][1])
            model_dict[novel_name] = score
            actual_novels.append(novel_name)
    novel_scores[model] = model_dict

import pdb; pdb.set_trace()

with open('novel_stats.txt', 'w') as o:
    o.write('Novel\t'\
            'Length in words\tLength in sentences\t'\
            'Number of entities\tStd of names mentions\t'\
            'BERT\tELMO\tGPT2\tWord2Vec\n')

csv_dict = {'novel' : list(), \
            'words' : list(), \
            'sentences' : list(), \
            'entities' : list(), \
            'mentions_std' : list(), \
            'bert' : list(), \
            'elmo' : list(), \
            'gpt2' : list(), \
            'w2v' : list(), \
            }

for path in paths:
    _, novel_words, max_sentences_length, max_entity_length, mentions_std = process_novel(path, get_vectors=False)
    novel_name = path[0]
    if novel_name in actual_novels:
        with open('novel_stats.txt', 'a') as o:
            o.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(novel_name, \
                                                          novel_words, max_sentences_length, \
                                                          max_entity_length, mentions_std, \
                                                          novel_scores['bert'][novel_name], \
                                                          novel_scores['elmo'][novel_name], \
                                                          novel_scores['gpt2_pearson_r'][novel_name], \
                                                          novel_scores['w2v_cosine_similarity'][novel_name], \
                                                          )
                   )

        csv_dict['novel'].append(novel_name)
        csv_dict['words'].append(novel_words)
        csv_dict['sentences'].append(max_sentences_length)
        csv_dict['entities'].append(max_entity_length)
        csv_dict['mentions_std'].append(mentions_std)
        csv_dict['bert'].append(novel_scores['bert'][novel_name])
        csv_dict['elmo'].append(novel_scores['elmo'][novel_name])
        csv_dict['gpt2'].append(novel_scores['gpt2_pearson_r'][novel_name])
        csv_dict['w2v'].append(novel_scores['w2v_cosine_similarity'][novel_name])

data_frame = pandas.DataFrame.from_dict(csv_dict)
data_frame.to_csv('novel_by_novel_scores.csv')
