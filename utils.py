import collections
import itertools
import math
import numpy
import os
import re
import random
import scipy
import multiprocessing

from tqdm import tqdm

def load_model(model_name):

    if model_name == 'bert':
        from transformers import BertTokenizer, BertModel
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        model = [tokenizer, bert_model]

    if model_name == 'gpt2':
        from transformers import GPT2Tokenizer, GPT2Model
        import torch
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', additional_special_tokens=['CHAR'])
        gpt2_model = GPT2Model.from_pretrained('gpt2')
        model = [tokenizer, gpt2_model]

    elif model_name == 'elmo':
        import allennlp
        from allennlp.commands.elmo import ElmoEmbedder
        model = ElmoEmbedder()

    elif model_name == 'w2v':
        import gensim
        from gensim.models import Word2Vec
        model = Word2Vec.load('data/w2v_background_space/wiki_w2v_2018_size300_window5_max_final_vocab250000_sg1')

    return model

def get_novels_paths(novel_aficionados_path):

    all_paths = collections.defaultdict(list)

    novels_path = os.path.join(novel_aficionados_path, 'novels')
    for novel_folder in os.listdir(novels_path):
        current_novel_path = os.path.join(novels_path, novel_folder, 'test_files')
        current_files = os.listdir(current_novel_path)
        for f in current_files:
            if 'characters' in f:
                all_paths[novel_folder].append(f)
                number = re.sub('characters_list_|\.txt', '', f)
        all_paths[novel_folder].append('matched_common_nouns_list_{}.txt'.format(number))
        all_paths[novel_folder].append('novel/matched_{}_ready_for_replication.txt'.format(number))
        all_paths[novel_folder] = [os.path.join(current_novel_path, f) for f in all_paths[novel_folder]]
    return all_paths

def get_word_vectors(sentence, entity, model, model_name):

    ### Preparing the sentence
    entity_mask = '[MASK]' if model_name == 'bert' else 'CHAR'
    sentence = sentence.replace(entity, entity_mask)
    sentence = re.sub('\#|\$', '', sentence)
    sentence = re.sub('_', ' ', sentence)

    ### Getting word vectors
    if model_name == 'bert' or model_name == 'gpt2':
        word_vectors = huggingface(sentence, model)
    elif model_name == 'elmo':
        word_vectors = elmo(sentence, model)
    elif model_name == 'w2v':
        word_vectors = w2v(sentence, model)

    return word_vectors

def huggingface(sentence, model):

    tokenizer = model[0]
    model = model[1]
    model_name = vars(model)['name_or_path']

    input_ids = tokenizer(sentence, return_tensors='pt')
    readable_input_ids = input_ids['input_ids'][0].tolist()
    if model_name == 'bert-base-uncased':
        tokens = [tokenizer._convert_id_to_token(id_num) for id_num in readable_input_ids]
        relevant_indices = [i for i, input_id in enumerate(readable_input_ids) if input_id == 103] 
    else:
        tokens = [tokenizer._convert_id_to_token(id_num) for id_num in readable_input_ids]
        relevant_indices = [i for i, input_token in enumerate(tokens) if 'CHAR' in input_token] 

    assert len(relevant_indices) >= 1

    if len(tokens) != len(readable_input_ids):
        #import pdb; pdb.set_trace()
        print('Issue with tokenization')

    outputs = model(**input_ids, return_dict=True, output_hidden_states=True, output_attentions=False)

    assert len(readable_input_ids) == len(outputs['hidden_states'][1][0])
    word_vectors = list()

    for i in relevant_indices:
        layer_container = list()
        ### Using the first 4 layers in BERT
        for layer in range(1, 5):
            layer_container.append(outputs['hidden_states'][layer][0][i].detach().numpy())
        layer_container = numpy.average(layer_container, axis=0)
        assert len(layer_container) == 768
        word_vectors.append(layer_container)
    return word_vectors

def elmo(sentence, model):

    sentence = sentence.split()
    relevant_indices = [i for i, input_token in enumerate(sentence) if input_token == 'CHAR'] 
    ### Using the 2nd layer (the top layer) in ELMO
    full_sentence = model.embed_sentence(sentence)[2]
    word_vectors = [full_sentence[i] for i in relevant_indices]

    return word_vectors

def w2v(sentence, model):

    sentence = [w for w in sentence.split()]
    entity_indices = [i for i, w in enumerate(sentence) if w == 'CHAR']
    ### Using an average of the vectors contained in a window of size 2 (2 to the right and 2 to the left as the 
    ### representation for the word, as done in Word2Vec's CBOW algorithm
    word_vectors = list()
    for i in entity_indices:
        lower_bound = max(0, i-2)
        upper_bound = min(len(sentence), i+2+1)
        current_window = [sentence[k] for k in range(lower_bound, i)] + [sentence[k] for k in range(i+1, upper_bound)]
        word_vocabs = [model.wv.vocab[w] for w in current_window if w in model.wv.vocab]
        if len(word_vocabs) >= 1:
            word2_indices = [word.index for word in word_vocabs]
            #word_vector = predict_output_w2v_embedding(model, word2_indices)
            bow_mean = numpy.average(model.wv.vectors[word2_indices], axis=0)
            word_vectors.append(bow_mean)

    return word_vectors

def get_scores(vectors, final_entities, chosen_metric):

    ranking_scores = list()
    norm_ranking_scores = list()
    distance = DistanceMeasure(chosen_metric)

    for part, part_vecs in vectors.items():
        for e in final_entities:
            query_vector = part_vecs[e]
            target_vecs = vectors['a'] if part == 'b' else vectors['b']
            results = collections.defaultdict(float)
            for target_e, target_vec in target_vecs.items():
                results[target_e] = distance.compute(query_vector, target_vec)
            results = [k for k in sorted(results.items(), key=lambda item : item[1], reverse=True)]
            ranking = [i+1 for i, v in enumerate(results) if v[0] == e]
            assert len(ranking) == 1
           
            #normalized_ranking_old = (ranking[0] - len(results)) / (1 - len(results))
            normalized_ranking = (ranking[0] -1) / (len(results) - 1)

            ranking_scores.append(ranking[0])
            norm_ranking_scores.append(normalized_ranking)
            #print('{}\t{}'.format(normalized_ranking, ranking)) 

    if len(final_entities) > 2:
        rsa_combs = [c for c in itertools.combinations(final_entities, 2)]
        list_one = [distance.compute(vectors['a'][e_one], vectors['a'][e_two]) for e_one, e_two in rsa_combs]
        list_two = [distance.compute(vectors['b'][e_one], vectors['b'][e_two]) for e_one, e_two in rsa_combs]
        rsa_score = distance.compute(list_one, list_two)
    else:
        rsa_score = numpy.nan

    return ranking_scores, norm_ranking_scores, rsa_score

class DistanceMeasure:

    def __init__(self, chosen_metric):
        self.available_metrics = ['pearson_r', 'spearman_rho', 'cosine_similarity']
        if chosen_metric in self.available_metrics:
            self.chosen_metric = chosen_metric
        else:
            print('Chosen metric ({}) not found amongst the ones available. Falling back to cosine_similarity'.format(chosen_metric))
            self.chosen_metric = 'cosine_similarity' 

    def compute(self, vector_one, vector_two):

        if len(vector_one) != len(vector_two):
            raise ValueError('Vectors must be of same length: here they have {} vs {} dimensions'.format(len(vector_one), len(vector_two)))

        if self.chosen_metric == 'cosine_similarity':
            return self.cosine_similarity(vector_one, vector_two)
        elif self.chosen_metric == 'pearson_r':
            return self.pearson_r(vector_one, vector_two)
        elif self.chosen_metric == 'spearman_rho':
            return self.spearman_rho(vector_one, vector_two)


    def cosine_similarity(self, vector_one, vector_two):

        num = numpy.dot(vector_one, vector_two)
        den_one = numpy.dot(vector_one, vector_one)
        den_two = numpy.dot(vector_two, vector_two)
        den = math.sqrt(den_one) * math.sqrt(den_two)
        cos = num /den

        return cos

    def pearson_r(self, vector_one, vector_two):

        r = scipy.stats.pearsonr(vector_one, vector_two)[0]

        return r
        
    def spearman_rho(self, vector_one, vector_two):

        rho = scipy.stats.spearmanr(vector_one, vector_two)[0]

        return rho

#def sentence_permutation(vectors_per_sentence, max_sentences_length):
def sentence_permutation(input_tuple):

    vectors_per_sentence = input_tuple[0]
    max_sentences_length = input_tuple[1]
    chosen_metric = input_tuple[2]

    results_collector = list()

    randomized_sentence_indices = random.sample(vectors_per_sentence.keys(), k=len(vectors_per_sentence.keys()))[:max_sentences_length]
    ### Slicing sentences in two
    break_point = int(len(randomized_sentence_indices)/2)
    parts = collections.defaultdict(list)
    parts['a'] = [vectors_per_sentence[i] for i in randomized_sentence_indices[:break_point]]
    parts['b'] = [vectors_per_sentence[i] for i in randomized_sentence_indices[break_point:]]
    
    ### Collecting vectors

    vectors = collections.defaultdict(dict)
    std_collector = 0.0

    for part, sentence_dics in parts.items():
        part_vectors = collections.defaultdict(list)
        for sentence_dic in sentence_dics:
            for e, vec in sentence_dic.items():
                part_vectors[e].append(vec)
        ### Collecting the standard deviation of the number of sentences per-entity
        #statistics_collector[novel_name]['standard deviation of number of per-entity sentences'] += numpy.nanstd([len(v) for k, v in part_vectors.items()])
        std_collector += numpy.nanstd([len(v) for k, v in part_vectors.items()])
        ### Averaging vectors for each entity within a part so that, in the end, each part will be represented by only one entity vector
        part_vectors = {k : numpy.average(v, axis=0) for k, v in part_vectors.items()}
        vectors[part] = part_vectors

    final_entities = [k for k in vectors['a'].keys() if k in vectors['b'].keys()]
    if len(final_entities) > 2:
        ranking_scores, norm_ranking_scores, rsa_score = get_scores(vectors, final_entities, chosen_metric)
        
        #category_scores[category]['pure ranking'].append(ranking_scores)
        #category_scores[category]['normalized ranking'].append(norm_ranking_scores)
        #category_scores[category]['rsa score'].append(rsa_score)
        '''
        print('Category: {}\n\n'
                '\tMean score: {}\n'
                '\tMedian score: {}\n'
                '\tStandard deviation: {}\n'
                '\tRSA score: {}\n'
                ''.format(category, numpy.nanmean(ranking_scores), numpy.nanmedian(ranking_scores), numpy.nanstd(ranking_scores), rsa_score))
        '''

        return [std_collector, ranking_scores, norm_ranking_scores, rsa_score]

def process_novel(novel_tuple):


    current_stats = collections.defaultdict(float)

    novel_name = novel_tuple[0]
    novel_paths = novel_tuple[1]
    model_name = novel_tuple[2]
    model = novel_tuple[3]
    chosen_metric = novel_tuple[4]
    ### Extracting the characters' list
    characters_file = [l.strip().split('\t') for l in open(novel_paths[0])]
    characters = [c[1].split('_')[0] for c in characters_file if int(c[0]) >= 10]

    ### Extracting the common nouns' list
    common_nouns_file = [l.strip().split('\t') for l in open(novel_paths[1])]
    common_nouns = [c[1].split('_')[0] for c in common_nouns_file if int(c[0]) >= 10]

    ### Trimming the lists to the minimum possible length
    max_entity_length = min(len(characters), len(common_nouns))
    characters = characters[:max_entity_length]
    common_nouns = common_nouns[:max_entity_length]
    current_stats['number of entities considered'] = float(max_entity_length)

    ### Preparing the novel
    novel = collections.defaultdict(list)
    novel_file = [l.strip() for l in open(novel_paths[2])]
    novel['characters'] = [re.sub('#', '', l) for l in novel_file if '$' in l]
    novel['common nouns'] = [re.sub('$', '', l) for l in novel_file if '#' in l]
    
    ### Storing the maximum length per entity
    max_sentences_length = min([len(v) for k, v in novel.items()])
    current_stats['number of sentences considered'] = float(max_sentences_length)

    novel_sentences = collections.defaultdict(dict)

    ### Testing
    for category, sentences in novel.items():

        #vectors_per_sentence = collections.defaultdict(lambda : collections.defaultdict(list))
        vectors_per_sentence = collections.defaultdict(dict)

        for sentence_number, s in enumerate(sentences):
            ### Finding which entities are present in this sentence
            present_entities = [e for e in s.split() if '$' in e or '#' in e]
            ### For each entities, obtaining the vectors
            for e in present_entities:
                entity_vectors = get_word_vectors(s, e, model, model_name)  
                if len(entity_vectors) >= 1:
                    vectors_per_sentence[sentence_number][e] = numpy.average(entity_vectors, axis=0)

        novel_sentences[category] = vectors_per_sentence

    return novel_sentences, max_entity_length, max_sentences_length



    '''
    output_results = dict()
    output_stats = dict()

    for category, all_scores in category_scores.items():
        for score_type, scores in all_scores.items():
            #final_results[category][novel_name][score_type] = scores
            output_results[(category, novel_name, score_type)] = scores

    for category, all_stats in statistics_collector.items():
        for score_type, score in all_stats.items():
            output_stats[(category, novel_name, score_type)] = score

    return [output_stats, output_results]
    '''
