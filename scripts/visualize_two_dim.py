from sklearn.manifold import TSNE
import numpy as np
import re
#import torch
#import dill as pickle
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import tqdm
import os
import itertools

from tqdm import tqdm
from matplotlib import rcParams
from matplotlib import font_manager as font_manager
from collections import defaultdict

def tsne_plot_words(title, words, embeddings, colors, filename=None):
    plt.figure(figsize=(16, 9))
    for embedding, word in zip(embeddings, words):
        #x = embeddings[:, 0]
        #y = embeddings[:, 1]
        #import pdb; pdb.set_trace()
        plt.scatter(embedding[0], embedding[1], c=(colors[word].reshape(1, colors[word].shape[0])), alpha=1, edgecolors='k', s=120)
        plt.annotate(word, alpha=1, xy=(embedding[0], embedding[1]), xytext=(10, 7), textcoords='offset points', ha='center', va='bottom', size=12)
    #plt.legend(loc=4)
    #plt.title(title, fontdict={'fontsize': 24, 'fontweight' : 'bold', 'color' : rcParams['axes.titlecolor'], 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}, pad=10.0)
    plt.title(title, fontsize='xx-large', fontweight='bold', pad = 15.0)
    #plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')

def cleanup(version, dictionary, mode='doppelganger'):
    name_selection = []
    second_label = 'b' if mode == 'doppelganger' else 'wiki'
    for name, vector in dictionary.items():
        if re.sub('_a$', '_{}'.format(second_label), name) in dictionary.keys() and re.sub('_{}$'.format(second_label), '_a', name) in dictionary.keys():
            if name not in name_selection:
                name_selection.append(name)
    cleaned_up = {name : dictionary[name] for name in name_selection} 
    if version == 'count':
        '''
        if mode == 'doppelganger':
            cleaned_up_copy = cleaned_up.copy()
            cleaned_up = {k : v.reshape(v.shape[1],) for k, v in cleaned_up_copy.items()}
        '''
        '''
        dimension_cutoff = min([v.shape[0] for k, v in cleaned_up.items()])
        cleaned_up_copy = cleaned_up.copy()
        cleaned_up = {k : v[:dimension_cutoff] for k, v in cleaned_up_copy.items()}
        '''
    return cleaned_up

def merge_two_dicts(version, dict_doppelganger, dict_one):
    '''
    if version == 'count':
        dop = dict_doppelganger.copy()
        dict_doppelganger = {k : v.reshape(v.shape[1],) for k, v in dop.items()}
        one = dict_one.copy()
        dict_one = {k : v.reshape(v.shape[1],) for k, v in one.items()}
    '''
    z = dict_doppelganger.copy()
    z_clean = {k : v for k, v in z.items() if k[-1] == 'a'}
    z_clean.update(dict_one)
        
    return z_clean

def prepare_damn_numpy_arrays(dictionary):
    new_dict = defaultdict(np.ndarray)
    for k, v in dictionary.items():
        if (v.shape)[0] == 1:
            new_dict[k] = v.reshape(v.shape[1])
        else:
            new_dict[k] = v
    return(new_dict)


def get_colors_dict(test, proper_names, common_nouns):
    #colors_gen = cm.prism(np.linspace(0, 1, (len(names)*2)))
    color_dict = defaultdict(np.ndarray)
    collection = {'proper' : [k for k, v in proper_names.items()], 'common' : [k for k, v in common_nouns.items()]}
    for category, content in collection.items():
        if category == 'proper': 
            colors_gen = cm.Wistia(np.linspace(0, 1, (len(proper_names.keys()))))
        if category == 'common':
            colors_gen = cm.winter(np.linspace(1, 0, (len(common_nouns.keys()))))
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

def plot_lines(output_folder, models, proper_list, common_list, title='', x_axis='', y_axis=''):
    plt.plot(models, proper_list, label='Proper names', marker='o', mec='k', mfc='white', )
    for a, k in zip(models, proper_list):
        plt.annotate(round(k, 2), (a, k), textcoords='offset points', xytext=(0,10), ha='center')
    plt.plot(models, common_list, label='Common nouns', marker='o', mec='k', mfc='white')
    for a, k in zip(models, common_list):
        plt.annotate(round(k, 2), (a, k), textcoords='offset points', xytext=(0,10), ha='center')
    plt.xticks(models, rotation = 45)
    plt.title(title, fontsize='xx-large')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.tight_layout(pad=3.0)

    plt.savefig(os.path.join(output_folder, 'line_plot.png'), transparent=True, dpi=600)

font_dirs = ['/import/cogsci/andrea/fonts/helvetica_ltd', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
#[f.name for f in matplotlib.font_manager.fontManager.ttflist]

matplotlib.rcParams['font.family'] = 'Helvetica LT Std'


tests = ['quality', 'doppelganger']
#tests = ['doppelganger']
versions = ['bert_base', 'bert_large', 'elmo', 'n2v', 'w2v', 'count']
#versions = ['bert']
combinations = []
for t in tests:
    for v in versions:
        if (t, v) not in combinations:
            combinations.append((t, v))

for combination in tqdm(combinations):

    test = combination[0]
    version = combination[1]

    os.makedirs('tSNE/{}'.format(test), exist_ok=True)
    #v = pickle.load(open('elmo_novels/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/244.pickle', 'rb'))
    pickle_proper = prepare_damn_numpy_arrays(pickle.load(open('{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/proper_names_matched/{}_test/244.pickle'.format(version, test), 'rb')))
    pickle_common = prepare_damn_numpy_arrays(pickle.load(open('{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/common_nouns_unmatched/{}_test/244.pickle'.format(version, test), 'rb')))

    v_proper = cleanup(version, pickle_proper)
    v_common = cleanup(version, pickle_common)
    
    
    if test == 'quality':
        v_proper_merged = merge_two_dicts(version, prepare_damn_numpy_arrays(pickle.load(open('{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/proper_names_matched/doppelganger_test/244.pickle'.format(version), 'rb'))), v_proper)
        v_common_merged = merge_two_dicts(version, prepare_damn_numpy_arrays(pickle.load(open('{}_training/A_Study_in_Scarlet_by_Arthur_Conan_Doyle/common_nouns_unmatched/doppelganger_test/244.pickle'.format(version), 'rb'))), v_common)
        v_proper = cleanup(version, v_proper_merged, mode='wiki')
        v_common = cleanup(version, v_common_merged, mode='wiki')
    if version == 'count':
        minimum_cutoff = min([v.shape[0] for k, v in v_proper.items()] + [v.shape[0] for k, v in v_common.items()])
        proper_copy = v_proper.copy()
        common_copy = v_common.copy()
        v_proper = {k : v[:minimum_cutoff] for k, v in proper_copy.items()}
        v_common = {k : v[:minimum_cutoff] for k, v in common_copy.items()}
    #v_proper = pickle.load(open('{}_training/Pride_and_Prejudice_by_Jane_Austen_/proper_names_matched/{}_test/1342.pickle'.format(version, test), 'rb'))
    #v_common = pickle.load(open('{}_training/Pride_and_Prejudice_by_Jane_Austen_/common_nouns_unmatched/{}_test/1342.pickle'.format(version, test), 'rb'))
    #vectors = np.array([v[0].reshape(1,-1) for k, v in v.items()])

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
    #print([(v).shape for k, v in v_proper.items()])
    #print([(v).shape for k, v in v_common.items()])
    #import pdb; pdb.set_trace()
    embeddings_en_2d = tsne_model_en_2d.fit_transform([v for k, v in v_proper.items()] + [v for k, v in v_common.items()])
    
    #% matplotlib inline
    tsne_plot_words('{} test - t-SNE visualization of the {} vectors for A Study in Scarlet'.format(test.capitalize(), version), [k for k in v_proper.keys()] + [k for k in v_common.keys()], embeddings_en_2d, colors_dict, 'tSNE/{}/{}_study_scarlet.png'.format(test, version))
    #tsne_plot_words('t-SNE visualization of the {} vectors for Pride and Prejudice'.format(version), names, embeddings_en_2d, color_dict, 'tSNE/{}/{}_pnp.png'.format(test, version))
