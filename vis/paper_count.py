import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

from info import jourven_list, paths

sns.set()

all_color_cylce = ['lawngreen',
                   'forestgreen',
                   'navy',
                   'deepskyblue',
                   'sienna',
                   'sandybrown',
                   'brown',
                   'yellow',
                   'lightyellow',
                   'violet',
                   'darkorchid'
                   ]

nlp_model = "en_newmodel"

path_to_fig_save = "/home/norpheo/Documents/thesis/Results/fig/stats"
df = pd.read_pickle(os.path.join(paths.to_root, 'annotations_version', nlp_model, 'info_db.pandas'))

print(len(df))
older = df['year'] >= 1990
younger = df['year'] <= 2017
df2 = df[older & younger]
print(len(df2))

journals = list()
venues = list()

for region in jourven_list.venues:
    vr = jourven_list.venues[region]
    for mag in vr:
        venues.append(mag)

for region in jourven_list.journals:
    vr = jourven_list.journals[region]
    for mag in vr:
        journals.append(mag)

all_year_dict = dict()
for year in range(1948, 2019):
    all_year_dict[year] = 0

category_ts = dict()
for category in jourven_list.journals:
    category_ts[category] = dict()
    for year in range(1948, 2019):
        category_ts[category][year] = 0

abs_cats = dict()

for abstract_id, entry in df.iterrows():
    year = entry['year']
    journal = entry['journal']
    venue = entry['venue']

    if journal in journals or venue in venues:
        all_year_dict[year] += 1

    for cat in category_ts:
        if journal in jourven_list.journals[cat] or venue in jourven_list.venues[cat]:
            category_ts[cat][year] += 1

            if abstract_id not in abs_cats:
                abs_cats[abstract_id] = set()

            abs_cats[abstract_id].add(cat)

cat_n_grams_list = list()
for aid in abs_cats:
    ngram = ", ".join(abs_cats[aid])
    cat_n_grams_list.append(ngram)

cat_n_grams_set = set(cat_n_grams_list)
print(len(cat_n_grams_set))
for ngram in cat_n_grams_set:
    print(f"{ngram}: {cat_n_grams_list.count(ngram)}")

"""
    Plot for every Category
"""

# fig = plt.figure(figsize=(12, 8))
# fig.subplots_adjust(hspace=0.1, wspace=0.4)
ax = [None, None, None, None, None, None]
fig, ((ax[0], ax[1], ax[2]), (ax[3], ax[4], ax[5])) = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
print("PLOTTING:")
stacked_y = list()
stacked_labels = list()
for i, cat in enumerate(category_ts):
    #ax = fig.add_subplot(2, 3, i+1)

    year_dict = category_ts[cat]

    category_string = cat.replace(" ", "")

    years_list = [key for key in year_dict.keys() if key < 2018]
    years_list.sort()

    x = np.array(years_list)
    y = np.array([year_dict[year] for year in years_list])
    stacked_y.append(y)
    stacked_labels.append(cat)

    #fig, ax = plt.subplots()
    ax[i].bar(x, y, width=1.0, facecolor=all_color_cylce[i], edgecolor=all_color_cylce[i])
    # ax.bar(x, y, bar_width, alpha=opacity, color='b')
    start, end = ax[i].get_xlim()
    # start = int(start)
    start = 1970
    end = int(end)
    ax[i].xaxis.set_ticks(np.arange(start, end, 10))
    ax[i].set_title(label="\n".join(wrap(cat, 25)), fontdict={'fontsize': 12,
                                                              'fontweight': 'normal',
                                                              'verticalalignment': 'baseline',
                                                              'horizontalalignment': 'center'})
    ax[i].set_xlim(start, end)

ax[3].set_xlabel("years")
ax[4].set_xlabel("years")
ax[5].set_xlabel("years")
ax[0].set_ylabel("number of publications")
ax[3].set_ylabel("number of publications")

# years_list = [key for key in all_year_dict.keys() if key < 2018]
# years_list.sort()
# x = years_list
# y = [all_year_dict[year] for year in years_list]
#
# ax[i+1].plot(x, y)
#
# start, end = ax[i+1].get_xlim()
# start = 1970
# end = int(end)
# ax[i+1].xaxis.set_ticks(np.arange(start, end, 10))
# ax[i+1].set_title(label="All publications", fontdict={'fontsize': 12,
#                                                       'fontweight': 'bold',
#                                                       'verticalalignment': 'baseline',
#                                                       'horizontalalignment': 'center'})
# ax[i+1].set_xlim(start, end)


plt.savefig(os.path.join(path_to_fig_save, f"fig_pub_count.png"), bbox_inches="tight")
# plt.show()


"""
    Plot stacked Bar
"""
color_cycle = all_color_cylce[:len(stacked_y)]

stacked_fig, stacked_ax = plt.subplots()
stacked_ax.stackplot(x, stacked_y,
                     labels=stacked_labels,
                     colors=color_cycle)

start, end = stacked_ax.get_xlim()
start = 1970
end = int(end)
stacked_ax.xaxis.set_ticks(np.arange(start, end, 10))

# stacked_ax.set_title(label="All publications - Stacked", fontdict={'fontsize': 12,
#                                                                    'fontweight': 'normal',
#                                                                    'verticalalignment': 'baseline',
#                                                                    'horizontalalignment': 'center'})
stacked_ax.legend(loc='upper left')
stacked_ax.set_xlim(start, end)
stacked_ax.set_xlabel("years")
stacked_ax.set_ylabel("number of publications")
plt.savefig(os.path.join(path_to_fig_save, f"fig_stacked_pub_count.png"), bbox_inches="tight")
# plt.show()