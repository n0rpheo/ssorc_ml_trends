import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np

from pop_tracking.mod3_make_timeseries import smooth_dataframe, get_trend_dataframe

from info import paths


def get_features(input_df):
    norm_df = input_df.div(input_df.sum(axis=1), axis=0)

    time_range = norm_df.index.values
    tr_len = len(time_range)
    start_range = time_range[:int(tr_len/3)]
    # mid_range = time_range[int(tr_len/3):int(2*tr_len/3)]
    end_range = time_range[int(2*tr_len/3):]

    features = list()
    for column in norm_df:
        ldp_start = norm_df[column][start_range].mean()
        ldp_end = norm_df[column][end_range].mean()
        ldp_all = norm_df[column][time_range].mean()

        features.append(ldp_start)
        features.append(ldp_all)
        features.append(ldp_end)
        features.append((ldp_end / ldp_start) if ldp_start > 0 else 0)
    return features


sns.set()

nlp_model = "en_newmodel"

v_num_topics = 500

per_year_fn = f"pop_per_year_p2_{v_num_topics}.pickle"
per_abstract_fn = f"pop_per_abstract_p2_{v_num_topics}.pickle"
tm_info_fn = f"lemma_vec_occ2_{v_num_topics}.info"

# RF Labels
rf_labelnames_dict_fn = f"ann_model_labelnames.dict"

# OUTPUT
prefix = ""

# Year Range
year_start = 1980
year_end = 2017

# All Paths
path_to_tm = os.path.join(paths.to_root, "topic_modeling", nlp_model)
path_to_popularities = os.path.join(paths.to_root, "popularities", nlp_model)
path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)
path_to_fig_save = "/home/norpheo/Documents/thesis/Results/fig/popularities/new_model/topic_trends"

with open(os.path.join(path_to_tm, tm_info_fn), "rb") as handle:
    tm_info = pickle.load(handle)

with open(tm_info["model_path"], "rb") as handle:
    lda_model = pickle.load(handle)

with open(os.path.join(path_to_popularities, per_year_fn), 'rb') as handle:
    lda_per_year = pickle.load(handle)

with open(os.path.join(path_to_popularities, per_abstract_fn), 'rb') as handle:
    lda_per_abstract = pickle.load(handle)

with open(os.path.join(path_to_rfl, rf_labelnames_dict_fn), "rb") as handle:
    rflabel_dict = pickle.load(handle)

data = lda_per_year

year_range = range(year_start, year_end)

num_topics = lda_model.components_.shape[0]

df_list = list()

for topic in range(num_topics):

    year_list = list()
    popularity_list = dict()
    for rf_id in data[year_start]:
        rf_label = rflabel_dict['id2label'][rf_id]

        popularity_list[rf_label] = list()

    for year in year_range:
        if year not in data:
            continue
        year_list.append(year)

        for rf_id in data[year]:
            rf_label = rflabel_dict['id2label'][rf_id]

            popularity_list[rf_label].append(data[year][rf_id][0][topic])
    df = pd.DataFrame(popularity_list, index=list(year_range))
    df_list.append(df)


for topic_id, df_ori in enumerate(df_list):
    df = pd.DataFrame(df_ori.sum(axis=1), columns=['popularity'])

    df_losses = smooth_dataframe(df, smooth_column='popularity')
    df_trend = get_trend_dataframe(df_losses, trend_column='smoothed popularity')

    fig_plot, axes_plot = plt.subplots(1, 1, figsize=(7, 7), sharex=True, dpi=120)

    df.reset_index(inplace=True)
    df.columns = ['year', 'popularity']

    # df.plot(ax=axes_plot, color='k', kind='line')
    df.plot.scatter(x='year', y='popularity', s=10, c="DarkBlue", ax=axes_plot)
    fig_plot.suptitle(f'Popularity for Topic {topic_id}', y=0.95, fontsize=14)
    plt.savefig(os.path.join(path_to_fig_save, f"{v_num_topics}", f"{prefix}topic_{topic_id}.png"),
                bbox_inches="tight")
    # plt.show()
    plt.close()

    fig_trends, axes_trends = plt.subplots(2, 1, figsize=(7, 7), sharex=True, dpi=120)
    df.plot.scatter(x='year', y='popularity', s=10, c="DarkBlue", ax=axes_trends[0])

    df_losses.plot(ax=axes_trends[0], color='b', title='Popularity', kind='line')
    df_trend.plot(ax=axes_trends[1], color='g', title='Trend', kind='line')

    axes_trends[1].set_ylim([-1.1, 1.1])
    axes_trends[1].yaxis.set_ticks(np.arange(-1, 2, 1))

    fig_trends.suptitle(f'Trend Analysis for Topic {topic_id}', y=0.95, fontsize=14)
    plt.savefig(os.path.join(path_to_fig_save, f"{v_num_topics}", f"{prefix}pop_topic_{topic_id}.png"),
                bbox_inches="tight")

    # plt.show()
    plt.close()

    print(topic_id)
