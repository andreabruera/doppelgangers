import os
import re
import spacy

from tqdm import tqdm

### Reading files

print('Now reading files and cleaning them')

sentences = list()

for root, direc, filez in os.walk('../novel_aficionados_dataset'):
    for f in filez:
        if 'test_files' in root and 'novel' in root and \
                  'unmatched' in f and 'replication' in f:
            with open(os.path.join(root, f)) as i:
                current_file = [l.strip() for l in i.readlines()]
            relevant_sents = [l for l in current_file \
                                 if '$' in l or '#' in l]
            sentences.extend(relevant_sents)

clean_sents = list()
for s in sentences:
    
    s = re.sub(r'#.+#', 'COMMON', s)
    s = re.sub(r'\$.+\$', 'PROPER', s)
    clean_sents.append(s)

### Spacy preprocessing

print('Loading Spacy')

big_model = spacy.load("en_core_web_sm")
#pos_list = ['ADJ', 'ADV', 'CCONJ', 'DET', 'NOUN', 'PRON', 'PROPN', 'VERB']

relevant_pos = ['NOUN', 'VERB']
window_size = [2, 5, 7]

counters = {w : {'PROPER' : {cat : dict() for cat in relevant_pos}, \
            'COMMON' : {cat : dict() for cat in relevant_pos}} for w in window_size}

print('Collecting frequencies')

for s in tqdm(clean_sents):

    sentence = [[token.lemma_, token.pos_, token.text] for token in big_model(s, disable = [ 'ner', 'textcat'])]
    relevant_indices = [w_i for w_i, w in enumerate(sentence) if w[2] in ['PROPER', 'COMMON']]
    assert len(relevant_indices) >= 1
    for rel_i in relevant_indices:
        cat = sentence[rel_i][2]
        for w_s in window_size:
            
            window = sentence[rel_i+1:min(len(sentence), rel_i+w_s)] + \
                     sentence[max(0, rel_i-w_s):rel_i] 
            #if not len(window) >= 1:
            for w in window:
                if w[1] in relevant_pos:
                    if w[0] not in counters[w_s][cat][w[1]].keys():
                        counters[w_s][cat][w[1]][w[0]] = 1
                    else:
                        counters[w_s][cat][w[1]][w[0]] += 1

print('Now writing to file')

results_path = 'pos_frequencies'
os.makedirs(results_path, exist_ok=True)

for w_s, v_one in counters.items():
    for cat, v_two in v_one.items():
        for rel_cat, v_three in v_two.items():
            sort_w = sorted(v_three.items(), key=lambda item : item[1], reverse=True)[:1000]
            current_f = os.path.join(results_path, \
                                'window_{}'.format(w_s), cat)
            os.makedirs(current_f, exist_ok=True)
            current_f = os.path.join(current_f, \
                     '{}.freqs'.format(rel_cat))
            with open(current_f, 'w') as o:
                for w, w_f in sort_w:
                    o.write('{}\t{}\n'.format(w, w_f))


