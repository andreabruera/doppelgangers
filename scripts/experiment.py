import os
import sys
import io
import re
import dill
import pickle 
import collections

import argparse
import logging
import logging.config

import math
import scipy
import numpy 
import torch
import tqdm

from tqdm import tqdm


#import nonce2vec.utils.config as cutils
#import nonce2vec.utils.files as futils
#from nonce2vec.models.BERT_for_novels import BERT_test
#from nonce2vec.utils.files import Samples
#from nonce2vec.utils.novels_utilities import *
#from nonce2vec.utils.count_based_models_utils import *
#from nonce2vec.utils.space_quality_test import *

from collections import defaultdict
from torch import Tensor
from scipy.sparse import csr_matrix

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch.nn import CosineSimilarity, PairwiseDistance

### Parser for the parameters to be passed.

parser = argparse.ArgumentParser()

# a shared set of parameters when using gensim
parser.add_argument('--num-threads', type=int, default=1,
                           help='number of threads to be used by gensim')
parser.add_argument('--alpha', type=float,
                           help='initial learning rate')
parser.add_argument('--neg', type=int,
                           help='number of negative samples')
parser.add_argument('--window', type=int,
                           help='window size')
parser.add_argument('--sample', type=float,
                           help='subsampling rate')
parser.add_argument('--epochs', type=int,
                           help='number of epochs')
parser.add_argument('--min-count', type=int,
                           help='min frequency count')

### NOVELS EDIT: added the dest for this argument, because it's needed for calling the novels model of N2V or the basic one

parser.add_argument('--on', required=True,
                         choices=['elmo', 'bert', 'count', 'w2v', 'n2v', 'pos'],
                         help='type of test data to be used')
parser.add_argument('--model', required=False,
                         dest='background',
                         help='absolute path to word2vec pretrained model')
parser.add_argument('--number', required=True, dest='number',
                         help='number of the novel')
### NOVELS EDIT: added this argument, which is needed for getting the right folder
parser.add_argument('--folder', required=False,
                         dest='folder',
                         help='absolute path to the novel folder')
#parser.add_argument('--write_to_file', required=False, default='False',
                         #action='store_true',
                         #help='specify whether it is needed to write a file with top similarities')
parser.add_argument('--train-with', required = False, default = 1,
                         choices=['exp_alpha'],
                         help='learning rate computation function')
parser.add_argument('--lambda', type=float, required = False, default = 70.0,
                         dest='lambda_den',
                         help='lambda decay')
### NOVELS_EDIT: added the destination with the '_' because it didn't seem to be present in the original code - and this messes everything up!
parser.add_argument('--sample-decay', type=float, required = False,
                         default = 1.1, dest='sample_decay',
                         help='sample decay')
### NOVELS_EDIT: added the destination with the '_' because it didn't seem to be present in the original code - and this messes everything up!
parser.add_argument('--window-decay', type=int, required = False, default = 3, 
                         dest='window_decay',
                         help='window decay')
### NOVELS_EDIT: added this for training on the wikipedia pages
parser.add_argument('--quality_test', action='store_true', required = False, default = False)
### NOVELS_EDIT: added this optional argument for testing the random selection of the sentences
parser.add_argument('--random_sentences', required=False, action='store_true', help='random_sentences: instead of picking sentences the original order it picks up sentences randomly')
parser.add_argument('--bert_layers', required = False, type = int, default = 2)
parser.add_argument('--men_dataset', dest='men_dataset', required=False, type=str)
parser.add_argument('--common_nouns', action ='store_true', required=False)
parser.add_argument('--prototype', action ='store_true', required=False)
parser.add_argument('--wiki', action ='store_true', required=False)
parser.add_argument('--vocabulary', type=str, required=False)
parser.add_argument('--window_size', type=int, required=False, default=15)
parser.add_argument('--top_contexts', type=int, required=False)
parser.add_argument('--weight', type=int, required=False)
parser.add_argument('--write_to_file', action ='store_true', required=False)
parser.add_argument('--matched', action ='store_true', required=False)

args = parser.parse_args()

def train_count_sentence(sentence, entity_name_and_part, vocabulary, word_cooccurrences, args):

    sentence_length = len(sentence)

    for sentence_current_word_index, current_word in enumerate(sentence):

        if current_word == 'CHAR':

            lower_bound = max(0, sentence_current_word_index - args.window_size)
            upper_bound = min(sentence_length, sentence_current_word_index + args.window_size)
            window = [n for n in range(lower_bound, upper_bound) if n != sentence_current_word_index]
            #print([sentence[window] for n in window])

            for position in window:

                other_word = sentence[position] 
                other_word_index = reduced_vocabulary[other_word]

                if other_word_index != 0:

                    word_cooccurrences[vocabulary_current_word_index][other_word_index] += 1

    return word_cooccurrences, final_vectors

def train_pos_sentence(sentence, entity_name_and_part, pos_cooccurrences, window_size = 5):

    sentence_length = len(sentence)

    window_sizes = [2,5,7,10]

    for window_size in window_sizes:

        for sentence_current_word_index, current_word in enumerate(sentence):

            if current_word[0] == 'CHAR':

                lower_bound = max(0, sentence_current_word_index - window_size)
                upper_bound = min(sentence_length, sentence_current_word_index + window_size)
                window = [n for n in range(lower_bound, upper_bound) if n != sentence_current_word_index]
                #print([sentence[window] for n in window])

                for position in window:

                    other_word = sentence[position] 
                    other_word_pos = other_word[1]

                    current_entity = entity_name_and_part[:-2]
                    current_entity_and_window = '{}_{}'.format(current_entity, window_size)

                    pos_cooccurrences[current_entity_and_window][other_word_pos] += 1

    return pos_cooccurrences

def cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = numpy.dot(peer_v, query_v)
    den_a = numpy.dot(peer_v, peer_v)
    den_b = numpy.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))

def ppmi(csr_matrix):
    """Return a ppmi-weighted CSR sparse matrix from an input CSR matrix."""
    logger.info('Weighing raw count CSR matrix via PPMI')
    words = sparse.csr_matrix(csr_matrix.sum(axis=1))
    contexts = sparse.csr_matrix(csr_matrix.sum(axis=0))
    total_sum = csr_matrix.sum()
    # csr_matrix = csr_matrix.multiply(words.power(-1)) # #(w, c) / #w
    # csr_matrix = csr_matrix.multiply(contexts.power(-1))  # #(w, c) / (#w * #c)
    # csr_matrix = csr_matrix.multiply(total)  # #(w, c) * D / (#w * #c)
    csr_matrix = csr_matrix.multiply(words.power(-1))\
                           .multiply(contexts.power(-1))\
                           .multiply(total_sum)
    csr_matrix.data = np.log2(csr_matrix.data)  # PMI = log(#(w, c) * D / (#w * #c))
    csr_matrix = csr_matrix.multiply(csr_matrix > 0)  # PPMI
    csr_matrix.eliminate_zeros()
    return csr_matrix

class BERT_test:
        
    def train(args, sentence, character, model, tokenizer):
        layers = args.bert_layers
        sentence.insert(0, '[CLS]')
        temporary_masked_index = [i for i, v in enumerate(sentence) if v == '[MASK]'][0]
        for word_index, word in enumerate(sentence): 
            if word == '[MASK]' and word_index != temporary_masked_index:
                del sentence[word_index]
        
        sentence = ' '.join(sentence)  
        tokenized_text = tokenizer.tokenize(sentence)
        masked_index = [i for i, v in enumerate(tokenized_text) if v == '[MASK]'][0]
        assert tokenized_text[masked_index] == '[MASK]' 
        other_indexes={token : index for index, token in enumerate(tokenized_text) if index>0}

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        model.eval()
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)
  

        if args.sum_CLS:
            layered_tensor = [word_tensor[0][0] for word_tensor in encoded_layers] 

        elif args.sum_only:
            summed_layer = defaultdict(torch.Tensor)
            layered_tensor = []
            full_tensor = [word_tensor[0] for word_tensor in encoded_layers]
            for layer in full_tensor:
                for word_index, word_vector in enumerate(layer):
                    if word_index != 0 and word_index != masked_index:
                        if len(summed_layer) == 0:
                            summed_layer = word_vector
                        elif word_index != masked_index:
                            summed_layer += word_vector
                layered_tensor.append(summed_layer)
            
        else:
            layered_tensor = [word_tensor[0][masked_index] for word_tensor in encoded_layers] 
        
        other_words_vectors = {other_word: encoded_layers[11][0][other_word_index] for other_word, other_word_index in other_indexes.items()}

        assert len(layered_tensor) == 12
        layered_tensor = torch.stack(layered_tensor)

        return layered_tensor, other_words_vectors

def test_on_novel(args):

    algorithm = args.on

    if algorithm == 'pos':
        import spacy
        big_model = spacy.load("en_core_web_sm")
        pos_list = ['ADJ', 'ADV', 'CCONJ', 'DET', 'NOUN', 'PRON', 'PROPN', 'VERB']

    elif algorithm == 'elmo':
        import allennlp
        from allennlp.commands.elmo import ElmoEmbedder
        elmo = ElmoEmbedder()

    elif algorithm == 'bert':
        from transformers import BertModel, BertTokenizer 
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True, output_attentions = True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', never_split=['[MASK]'])

    elif algorithm == 'w2v':
        import gensim
        from gensim.models import Word2Vec
        model = Word2Vec.load('w2v_background_space/wiki_w2v_2018_size300_window5_max_final_vocab250000_sg1')

    elif algorithm == 'n2v':
        import n2v
        from n2v import main
        background = 'w2v_background_space/wiki_w2v_2018_size300_window5_max_final_vocab250000_sg1'

    logging.info('Training on {}'.format(algorithm))

    folder = args.folder
    number = args.number

    tests = ['doppelganger_test', 'quality_test']
    categories = ['proper_names', 'common_nouns']
    frequency = ['matched', 'unmatched']

    test_setups1 = [[t, c, f] for t in tests for c in categories for f in frequency]
    test_setups = [l for l in test_setups1 if l != ['doppelganger_test', 'proper_names', 'unmatched'] and l != ['quality_test', 'proper_names', 'unmatched']]

    for setup in test_setups:

        test = setup[0]
        category = setup[1]
        frequency = setup[2]

        novel_path = '{}/novel/{}_{}_ready_for_replication.txt'.format(folder, frequency, number)
        wikipedia_page_path = '{}/wikipedia_page/{}_{}_ready_for_replication.txt'.format(folder, frequency, number) 
        characters_list_path = '{}/characters_list_{}.txt'.format(folder, number)
        common_nouns_list_path = '{}/{}_common_nouns_list_{}.txt'.format(folder, frequency, number)
        
        entity_dict = defaultdict(numpy.ndarray) 
        variance_check_dict = defaultdict(list)
        novel_versions = defaultdict(list)

        current_folder = '{}/{}_{}/{}'.format(folder, category, frequency, test)
        os.makedirs('{}/details'.format(current_folder),exist_ok=True)

        if test == 'doppelganger_test':

            with open(novel_path) as input_text:
                input_text = input_text.readlines()
                mid_point = int(len(input_text)/2)
                novel_versions['part_a'] = [l.strip('\n') for l in input_text[:mid_point]]
                novel_versions['part_b'] = [l.strip('\n') for l in input_text[mid_point:]]
                #novel_versions['full'] = [l.strip('\n') for l in input_text]

        elif test == 'quality_test':
            with open(wikipedia_page_path) as input_text:
                input_text = input_text.readlines()
                novel_versions['wiki'] = [l.strip('\n') for l in input_text]

        if algorithm == 'count':
            list_of_words = []
            for line in input_text:
                words = (line.strip()).split()
                for w in words:
                    if w not in list_of_words:
                        list_of_words.append(re.sub('\$|#', '', w))
                        
            
        proper_names_file = open(characters_list_path).readlines()
        proper_names_counter = 0
        current_entity_list = []

        for l in proper_names_file:
            l = l.strip('\n').split('\t')
            if int(l[0]) >= 10:
                proper_names_counter += 1
                if category != 'common_nouns':
                    aliases = l[1].split('_')
                    current_entity_list.append(re.sub('\W+', '_', aliases[0]).lower())

        if category == 'common_nouns':
            current_entity_list = []
            input_file = open(common_nouns_list_path).readlines()
            for l in input_file:
                l = l.strip('\n').split('\t')
                if int(l[0]) >= 10:
                    aliases = l[1].split('_')
                    current_entity_list.append(re.sub('\W+', '_', aliases[0]).lower())

        current_entity_list = current_entity_list[:proper_names_counter]

        if len(current_entity_list) > 1:

            novel_versions_keys=novel_versions.keys()

            print('Length of the proper names list: {}'.format(proper_names_counter))
            print('Length of the entity list: {}\n Entity list: {}\n'.format(len(current_entity_list), current_entity_list))

            if algorithm == 'count':
                print('Now loading the background space...')
                with open('count_models/count_wiki_2/count_wiki_2_cooccurrences.pickle', 'rb') as word_cooccurrences_file:
                    lambda_word_cooccurrences = dill.load(word_cooccurrences_file)
                word_cooccurrences=defaultdict(lambda: defaultdict(int))
                for first_key, second_dict in lambda_word_cooccurrences.items():
                    for second_key, count_value in second_dict.items():
                        word_cooccurrences[first_key][second_key]=count_value
                background_vectors_length = max(lambda_word_cooccurrences.keys())
                del lambda_word_cooccurrences
                print('Now loading the vocabulary...')
                with open('count_models/count_wiki/count_wiki_vocabulary_trimmed.pickle', 'rb') as vocabulary_file:
                    vocabulary = dill.load(vocabulary_file)
                list_of_entities_count = []
                #list_of_words = []
 
            if algorithm == 'pos':
                pos_cooccurrences = defaultdict(lambda: defaultdict(int))

            for path in novel_versions_keys:

                if 'part_a' in path:
                    part='a'
                elif 'part_b' in path:
                    part='b'
                else:
                    part='wiki'
            
                version=novel_versions[path]

                overall_training_counter = 0


                for entity in current_entity_list:

                    sentence_count = 0
                    variance_collector = []

                    entity_counter={}
                    
                    for key in novel_versions_keys:
                        entity_counter[key] = 0
                        if category == 'common_nouns':
                            entity_counter[key] = len([line for line in novel_versions[key] if '#{}#'.format(entity) in line or ' {} '.format(entity) in line])
                        else:
                            entity_counter[key] = len([line for line in novel_versions[key] if '${}$'.format(entity) in line]) 

                    entity_name_and_part='{}_{}'.format(entity, part)


                    if 0 not in [count for count_index, count in entity_counter.items()] and overall_training_counter <= proper_names_counter:


                        #print('Current entity: {} - for part: {}'.format(entity, part))

                        if category == 'common_nouns':
                            sent_list = [re.sub('#{}#|\s{}\s'.format(entity, entity), 'CHAR', line) for line in version if '#{}#'.format(entity) in line] 
                        else:
                            sent_list = [re.sub('\${}\$'.format(entity), 'CHAR', line) for line in version if '${}$'.format(entity) in line] 
                        sent_list = [re.sub('#|\$', '', line) for line in sent_list] 

                        if args.random_sentences == True:
                            indexes = numpy.random.choice(len(sent_list), len(sent_list), replace=False)
                        else:
                            indexes = numpy.arange(len(sent_list))

######### ELMO          
                        if algorithm == 'elmo':


                            for index in indexes:

                                sentence = sent_list[index]

                                if sentence_count == 0:

                                    background_space = defaultdict(numpy.ndarray)

                                sentence_count+=1

                                if sentence_count <= 50:

                                    sentence = sentence.split()
                                    relevant_indices = [i for i, w in enumerate(sentence) if w == 'CHAR']

                                ### NOVELS_EDIT: added the <50 condition in order to reduce training time and to give more balanced training to every entity.
                                    ### NOVELS NOTE: this is the part where the training happens
                                    
                                    #print(sentence)
                                    #print(relevant_indices)
                                    full_sentence = elmo.embed_sentence(sentence)[2]
                                    assert len(sentence) == len(full_sentence)

                                    for i_i, i in enumerate(relevant_indices):
                                        if sentence_count == 1 and i_i == 0:
                                            entity_dict[entity_name_and_part] = full_sentence[i]
                                        elif sentence_count > 1:
                                            #for layer_index, layer in enumerate(entity_dict[entity_name_and_part]):
                                            entity_dict[entity_name_and_part] = numpy.add(entity_dict[entity_name_and_part], full_sentence[i])
                                        variance_collector.append(full_sentence[i])
                                    
                                    for other_word, other_word_vector in zip(sentence, full_sentence):
                                        if other_word != 'CHAR':
                                            if other_word in background_space.keys():
                                                background_space[other_word] = numpy.add(background_space[other_word], other_word_vector)
                                            else:
                                                background_space[other_word] = other_word_vector

    ############## END OF ELMO

##### BERT
                        if algorithm == 'bert':

                            for index in indexes:

                                sentence = re.sub('CHAR', '[MASK]', sent_list[index])

                                tokenized_text = tokenizer.tokenize(sentence)

                                if sentence_count <= 50 and '[MASK]' in tokenized_text:

                                    if sentence_count == 0:

                                        background_space = defaultdict(numpy.ndarray)

                                    sentence_count+=1
                                    print(tokenized_text)

                                ### NOVELS_EDIT: added the <50 condition in order to reduce training time and to give more balanced training to every character.
                                    ### NOVELS NOTE: this is the part where the training happens
                                    layered_tensor, other_words_vectors = BERT_test.train(args, tokenized_text, model, tokenizer)    
                                    if sentence_count == 1:
                                        entity_dict[entity_name_and_part] = layered_tensor[11].detach().numpy()  
                                    elif sentence_count > 1:
                                        #for layer_index, layer in enumerate(char_dict[character_name_and_part]):
                                        entity_dict[entity_name_and_part] = numpy.add(entity_dict[entity_name_and_part], layered_tensor[11].detach().numpy())
                                    variance_collector.append(layered_tensor[11].detach().numpy())
                                    
                                    for other_word, other_word_vector in other_words_vectors.items():
                                        if other_word != '[MASK]':
                                            if other_word in background_space.keys():
                                                background_space[other_word] = numpy.add(other_word_vector, background_space[other_word])
                                            else:
                                                background_space[other_word] = other_word_vector
    ####### END OF BERT

#### BEGINNING OF W2V
                        
                        if algorithm == 'w2v':


                            for index in indexes:

                                sentence = [w for w in sent_list[index].split() if w != '']

                                if sentence_count == 0:

                                    background_space = defaultdict(numpy.ndarray)
                                    entity_dict[entity_name_and_part] = numpy.zeros(len(model.wv['home']))

                                sentence_count+=1

                                if sentence_count <= 50:
                                    
                                    for word in sentence:
                                        if word != 'CHAR':
                                            try:
                                                word_vector = model.wv[word]
                                                if word not in background_space.keys():
                                                    background_space[word] = word_vector
                                                entity_dict[entity_name_and_part] = numpy.add(entity_dict[entity_name_and_part], word_vector)
                                            except KeyError:
                                                pass

                                    variance_collector.append(entity_dict[entity_name_and_part])

######## BEGINNING N2V
                        if algorithm == 'n2v':
                          
                            data = []

                            for sentence_count, index in enumerate(indexes):
                                sentence = [w for w in sent_list[index].split() if w != '' and 'CHAR' in sent_list[index].split()]
                                if sentence_count <= 50:
                                    data.append(sentence)

                            for sentence_count, sentence in enumerate(data):

                                model = main.load_n2v_model(background, 'CHAR')
                                vocab_size = len(model.wv.vocab)
                                try:
                                    model.build_vocab([sentence], update=True)
                                    if sentence_count == 0:
                                        background_space = defaultdict(numpy.ndarray)
                                        entity_dict[entity_name_and_part] = numpy.zeros(len(model.wv['home']))

                                    model.train([sentence], total_examples=model.corpus_count, epochs=model.epochs)

                                    entity_dict[entity_name_and_part] = numpy.add(entity_dict[entity_name_and_part], model.wv['CHAR'])

                                    for word in sentence:  ### Creating the subspace of words to be used for explicit similarities
                                        if word != 'CHAR':
                                            try:
                                                word_vector = model.wv[word]
                                                if word not in background_space.keys():
                                                    background_space[word] = word_vector
                                            except KeyError:
                                                pass
                                    variance_collector.append(entity_dict[entity_name_and_part])
                                except RuntimeError:
                                    pass

######## BEGINNING COUNT

                        if algorithm == 'count':

                            list_of_entities_count.append(entity_name_and_part)

                            for index in indexes:

                                if sentence_count == 0:
                                    vocabulary[entity_name_and_part] = len(vocabulary)

                                sentence = [w for w in sent_list[index].split() if w != '']

                                sentence_count+=1

                                if sentence_count <= 50:

                                    word_cooccurrences = train_count_sentence(sentence, entity_name_and_part, vocabulary, word_cooccurrences, args)
######## BEGINNING POS

                        if algorithm == 'pos':


                            #list_of_entities_count.append(entity_name_and_part)

                            for index in indexes:

                                sentence = [[token.text, token.pos_] for token in big_model(sent_list[index], disable = [ 'ner', 'textcat'])]
                                if sentence_count <= 50 and 'CHAR' in [w[0] for w in sentence]:
                                    sentence_count+=1

                                    pos_cooccurrences = train_pos_sentence(sentence, entity_name_and_part, pos_cooccurrences)

                        '''
                        ### Checking for the variance within each entity's representation

                        if len(variance_collector) > 1:

                            variance_check_dict[entity_name_and_part] = variance_collector
                            cosine_within_novel = []

                            for vector_index, vector in enumerate(variance_collector):
                                for other_vector_index, other_vector in enumerate(variance_collector):
                                    if vector_index < other_vector_index:
                                        cosine_within_novel.append(cosine_similarity(vector, other_vector))

                            median_similarity_within_novel = numpy.nanmedian(cosine_within_novel)
                            std_within_novel = numpy.nanstd(cosine_within_novel)
                            print('Median within-novel similarity for {}: {}'.format(entity_name_and_part, median_similarity_within_novel))
                            print('Within-novel similarity standard deviation for {}: {}'.format(entity_name_and_part, std_within_novel))
                            print('Number of mentions used for {}: {}\n'.format(entity_name_and_part, len(variance_collector)))

                            if args.write_to_file:

                                with open('{}/{}/details/{}_within_novel.stats'.format(folder, test, entity_name_and_part), 'w') as within_novel_file:
                                    within_novel_file.write('Median within-novel similarity for\t{}:\t{}\n'.format(entity_name_and_part, median_similarity_within_novel))
                                    within_novel_file.write('Within-novel similarity standard deviation for\t{}:\t{}\n'.format(entity_name_and_part, std_within_novel))
                                    within_novel_file.write('Number of sentences used for\t{}:\t{}\n'.format(entity_name_and_part, len(variance_collector)))

                        else:
                            if args.write_to_file:
                                with open('{}/{}/details/{}_within_novel.stats'.format(folder, test, entity_name_and_part), 'w') as within_novel_file:
                                    if len(variance_collector) == 1:
                                        within_novel_file.write('Median within-novel similarity for\t{}:\tnot computable - only one sentence\n'.format(entity_name_and_part))
                        '''

                            ### Calculating top similarities

                        if algorithm != 'count' and algorithm != 'pos':

                            #if len(variance_collector) >= 1:
                            if sentence_count > 0:

                                overall_training_counter += 1
                                #print('Entity number {}'.format(overall_training_counter))
                                current_background_space = {other_word : cosine_similarity(other_vector, entity_dict[entity_name_and_part]) for other_word, other_vector in background_space.items() if '#' not in other_word}
                                top_similarities = sorted(current_background_space, key = current_background_space.get, reverse = True)[:99]
                                top_info = [(word, current_background_space[word]) for word in top_similarities]
                                #print('Most similar words in this space for {}: {}'.format(entity_name_and_part, top_info))
                                if args.write_to_file:
                                    with open('{}/details/{}.similarities'.format(current_folder, entity_name_and_part), 'w') as similarities:
                                        similarities.write('Top similarities for {}:\n\n{}'.format(entity_name_and_part, top_info))

                            else:
                                logging.info('No sentences found for {}...'.format(entity_name_and_part))
                                #import pdb; pdb.set_trace()

            if algorithm == 'count':

                ### Creating the sparse matrix containing the word vectors

                logging.info('Now creating the word vectors')

                rows_sparse_matrix = []
                columns_sparse_matrix = []
                cells_sparse_matrix = []

                for row_word_index in tqdm(word_cooccurrences): 
                    for column_word_index in word_cooccurrences[row_word_index]:
                        if column_word_index <= background_vectors_length:
                            rows_sparse_matrix.append(row_word_index)
                            columns_sparse_matrix.append(column_word_index)
                            current_cooccurrence = word_cooccurrences[row_word_index][column_word_index]
                            cells_sparse_matrix.append(current_cooccurrence)

                shape_sparse_matrix = len(vocabulary)

                logging.info('Now building the sparse matrix')

                sparse_matrix_cooccurrences = csr_matrix((cells_sparse_matrix, (rows_sparse_matrix, columns_sparse_matrix)), shape = (shape_sparse_matrix, shape_sparse_matrix))

                logging.info('Now applying ppmi to the matrix')

                ppmi_matrix = ppmi(sparse_matrix_cooccurrences)

                del sparse_matrix_cooccurrences

                background_space = defaultdict(numpy.ndarray)

                for w in list_of_words:
                    try:
                        word_index = vocabulary[w] 
                        background_space[w] = numpy.array(ppmi_matrix[word_index][0].todense())
                    except KeyError:
                        pass


                for e in list_of_entities_count:
                    try:
                        entity_index = vocabulary[e] 
                        entity_dict[e] = numpy.array(ppmi_matrix[entity_index][0].todense())
                    except KeyError:
                        logging.info('There has been a mistake with {}'.format(e))
                
                    current_background_space = {other_word : cosine_similarity(other_vector, entity_dict[e]) for other_word, other_vector in background_space.items() if '#' not in other_word}
                    top_similarities = sorted(current_background_space, key = current_background_space.get, reverse = True)[:99]
                    top_info = [(word, current_background_space[word]) for word in top_similarities]
                    #print('Most similar words in this space for {}: {}'.format(entity_name_and_part, top_info))
                    if args.write_to_file:
                        with open('{}/details/{}.similarities'.format(current_folder, e), 'w') as similarities:
                            similarities.write('Top similarities for {}:\n\n{}'.format(e, top_info))

            if algorithm == 'pos':
            
                entity_dict = defaultdict(list)
                for entity, other_dict in pos_cooccurrences.items():
                    entity_vector = numpy.zeros(len(pos_list))
                    for pos, freq in other_dict.items():
                        try:
                            entity_vector[pos_list.index(pos)] = int(freq)
                        except ValueError:
                            pass
                    entity_vector = entity_vector / numpy.linalg.norm(entity_vector)
                    entity_dict[entity] = entity_vector
                
            with open('{}/{}.pickle'.format(current_folder, number),'wb') as out:

                pickle.dump(entity_dict,out,pickle.HIGHEST_PROTOCOL) 

            '''
            with open('{}/{}.pickle'.format(current_folder, number),'rb') as part_a_in:
            #with open('{}/{}.pickle'.format(folder, number),'rb') as part_a_in:
                dop = pickle.load(part_a_in)
                current_entities = [k for k in entity_dict.keys()]
                for i in current_entities:
                    i = re.sub('_wiki', '_a', i)
                    if i in dop.keys():
                        entity_dict[i] = dop[i]

            #with open('{}/{}_variance_dict.pickle'.format(folder, number),'wb') as out_variance:        
                #pickle.dump(variance_check_dict,out_variance,pickle.HIGHEST_PROTOCOL) 
            
            evaluation = NovelsEvaluation()
            if test == 'doppelganger_test':
                evaluation.generic_evaluation(folder, entity_dict, wiki_novel = False)
            elif test == 'quality_test':
                evaluation.generic_evaluation(folder, entity_dict, wiki_novel = True)
            '''

        elif len(current_entity_list) <= 1:
            print('The novel {} has not enough entities so as to proceed with testing'.format(args.folder))

####

test_on_novel(args)
