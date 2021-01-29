import pickle
import matplotlib
import collections
import numpy

from matplotlib import pyplot

results = pickle.load(open('temp/final_results_w2v_cosine_similarity.pkl', 'rb'))[0]

to_be_plotted = collections.defaultdict(lambda : collections.defaultdict(list))

for mode, mode_dict in results.items():
    for novel, novel_dict in mode_dict.items():
        for res_type, res in novel_dict.items():
            if res_type != 'rsa score':
                to_be_plotted[res_type][mode].append([s for ss in res for s in ss])

for res_type, res_dict in to_be_plotted.items():
    
    fig, ax = pyplot.subplots()
    colors = {'common nouns' : 'teal', 'characters' : 'goldenrod'}
    labels = list()
    colors_plot = list()
    data = list()

    for mode, mode_results in res_dict.items():
        data.append([numpy.nanmean(v) for v in mode_results])
        labels.append(mode)
        colors_plot.append(colors[mode])

    ax.hist(data, bins=30, color=colors_plot, label=labels, edgecolor='white')

    ax.legend()
    pyplot.savefig('plot/{}.png'.format(res_type.replace(' ', '_')))
    pyplot.clf()
    pyplot.close()

