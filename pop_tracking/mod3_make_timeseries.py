import pickle
import os
import seaborn as sns
import pandas as pd
import numpy as np
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from statsmodels.nonparametric.smoothers_lowess import lowess

from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
import gensim

from utils.functions import Scoring

from info import paths


def get_features(input_df):
    norm_df = input_df.div(input_df.sum(axis=1), axis=0)

    time_range = norm_df.index.values
    tr_len = len(time_range)
    start_range = time_range[:int(tr_len/3)]
    mid_range = time_range[int(tr_len/3):int(2*tr_len/3)]
    end_range = time_range[int(2*tr_len/3):]

    features = list()
    for column in norm_df:
        ldp_start = norm_df[column][start_range].mean()
        ldp_end = norm_df[column][end_range].mean()
        ldp_mid = norm_df[column][mid_range].mean()
        ldp_all = norm_df[column][time_range].mean()

        features.append(0 if math.isnan(ldp_start) else ldp_start)
        features.append(0 if math.isnan(ldp_mid) else ldp_mid)
        features.append(0 if math.isnan(ldp_end) else ldp_end)
        features.append(0 if math.isnan(ldp_all) else ldp_all)
        features.append((ldp_end / ldp_start) if ldp_start > 0 else 0)
        features.append((ldp_mid / ldp_start) if ldp_start > 0 else 0)

    return features


def smooth_dataframe(input_df, smooth_column):
    input_df.rename(columns={smooth_column: 'value'}, inplace=True)

    df_smoothed = pd.DataFrame(lowess(input_df.value, np.arange(len(input_df.value)), frac=0.8)[:, 1],
                               index=input_df.index, columns=[f"smoothed {smooth_column}"])
    input_df.rename(columns={'value': smooth_column}, inplace=True)
    return df_smoothed


def get_trend_dataframe(input_smooth_df, trend_column, m=1.01):

    l_value = None
    v_list = list()

    for index, row in input_smooth_df.iterrows():
        c_value = row[trend_column]
        if l_value is None:
            v_list.append(0)
        else:
            if c_value > m * l_value:
                v_list.append(1)
            elif m * c_value < l_value:
                v_list.append(-1)
            else:
                v_list.append(0)
        l_value = c_value
    v_list[0] = v_list[1]

    df_trend = pd.DataFrame(v_list, index=input_smooth_df.index, columns=['trend'])
    return df_trend


def lowess_trends_and_features(input_df_list):
    target_vector_ = list()
    feature_vector_ = list()

    for df in input_df_list:
        df_pop = pd.DataFrame(df.sum(axis=1), columns=['popularity'])
        df_losses = smooth_dataframe(df_pop, 'popularity')
        df_trend = get_trend_dataframe(df_losses, 'smoothed popularity')
        last_trend = None
        trend_length = 0
        start_year = df_trend.index[0]
        for index, row in df_trend.iterrows():
            trend = row['trend']

            if last_trend is None or trend == last_trend:
                trend_length += 1
                end_year = index
            else:
                if trend_length > 6:
                    feature_vector_.append(get_features(df.loc[start_year:end_year]))
                    target_vector_.append(last_trend)
                trend_length = 0
                start_year = index
            last_trend = trend
        feature_vector_.append(get_features(df.loc[start_year:end_year]))
        target_vector_.append(last_trend)

    return feature_vector_, target_vector_


def df2mldf(input_data, tm_dict, lda_model_, years):

    # Manually Extract Information from MLRes.occ.show_clusters
    topic_cluster_names = [("aimlcluster105", "SVM"),
                           ("aimlcluster93", "PCA"),
                           ("aimlcluster108", "Decision Trees"),
                           ("aimlcluster38", "Naive Bayes"),
                           ("aimlcluster122", "Neural Network")
                           ]

    ml_dfs = dict()
    for topic_cluster_name in topic_cluster_names:
        topic_cluster_id = tm_dict.token2id[topic_cluster_name[0]]
        topics = list()

        # Gather TM-Prob Information of Cluster
        for topic, comp in enumerate(lda_model_.components_):
            norm_factor = np.sum(comp)
            tc_prob = comp[topic_cluster_id] / norm_factor
            topics.append((topic, tc_prob))

        rf_labels = sorted(list(input_data[years[0]].keys()))

        pop_list = dict()
        for rfid in rf_labels:
            rfl = rflabel_dict['id2label'][rfid]

            pop_list[rfl] = list()

        for year in years:
            if year not in input_data:
                continue

            for rfid in rf_labels:
                rfl = rflabel_dict['id2label'][rfid]

                pop_list[rfl].append(sum([input_data[year][rfid][0][topic_[0]]*topic_[1] for topic_ in topics]))

        mldf = pd.DataFrame(pop_list, index=list(year_range))
        ml_dfs[topic_cluster_name[1]] = mldf
    return ml_dfs


if __name__ == '__main__':
    sns.set()

    nlp_model = "en_newmodel"

    # RF Labels
    rf_labelnames_dict_fn = f"ann_model_labelnames.dict"

    v_num_topics = 500

    # TM Info
    tm_info_fn = f"lemma_vec_occ2_{v_num_topics}.info"

    # Calculation
    per_year_fn = f"pop_per_year_p2_{v_num_topics}.pickle"
    # per_sentence_fn = f"pop_per_sentence_p2_{v_num_topics}.pickle"

    # Year Range
    year_start = 1980
    year_end = 2017

    # All Paths
    path_to_tm = os.path.join(paths.to_root, "topic_modeling", nlp_model)
    path_to_popularities = os.path.join(paths.to_root, "popularities", nlp_model)
    path_to_rfl = os.path.join(paths.to_root, "rhet_func_labeling", nlp_model)

    with open(os.path.join(path_to_tm, tm_info_fn), "rb") as handle:
        tm_info = pickle.load(handle)

    with open(tm_info["model_path"], "rb") as handle:
        lda_model = pickle.load(handle)

    with open(os.path.join(path_to_popularities, per_year_fn), 'rb') as handle:
        lda_per_year = pickle.load(handle)

    # with open(os.path.join(path_to_popularities, per_sentence_fn), 'rb') as handle:
    #     lda_per_sentence = pickle.load(handle)

    with open(os.path.join(path_to_rfl, rf_labelnames_dict_fn), "rb") as handle:
        rflabel_dict = pickle.load(handle)

    tm_dictionary = gensim.corpora.Dictionary.load(tm_info["dic_path"])

    data = lda_per_year

    year_range = range(year_start, year_end)

    ml_dfs = df2mldf(data, tm_dictionary, lda_model, year_range)

    num_topics = lda_model.components_.shape[0]

    df_list = list()

    for topic in range(num_topics):
        popularity_list = dict()
        for rf_id in data[year_start]:
            rf_label = rflabel_dict['id2label'][rf_id]

            popularity_list[rf_label] = list()

        for year in year_range:
            if year not in data:
                continue

            for rf_id in data[year]:
                rf_label = rflabel_dict['id2label'][rf_id]

                popularity_list[rf_label].append(data[year][rf_id][0][topic])
        df = pd.DataFrame(popularity_list, index=list(year_range))
        df_list.append(df)

    feature_vector, target_vector = lowess_trends_and_features(df_list)

    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    # Print
    uniqueValues, occurCount = np.unique(target_vector, return_counts=True)
    print("Unique Values : ", uniqueValues)
    print("Occurrence Count : ", occurCount)

    learning_features, holdback_features, learning_targets, holdback_targets = train_test_split(feature_vector,
                                                                                                target_vector,
                                                                                                test_size=0.2,
                                                                                                random_state=42,
                                                                                                shuffle=True)

    # scoring = ["f1_macro", "f1_micro", "f1_weighted", "accuracy", "balanced_accuracy"]
    scoring = ["accuracy", "balanced_accuracy"]
    cv = 10

    test_depths = [2, 3, 4, 5, 10, 20, 50]

    # print("Test DT Depth")
    # for test_depth in test_depths:
    #     dtc = DecisionTreeClassifier(random_state=0, max_depth=test_depth)
    #     result_dt = cross_validate(dtc, learning_features, learning_targets, scoring=scoring, cv=cv, return_train_score=False)
    #     print()
    #     print(f"Results DT @ {test_depth}")
    #     for scorer in result_dt.keys():
    #         print(f"{scorer}: {result_dt[scorer].mean()}")

    dtc = DecisionTreeClassifier(random_state=0, max_depth=3)
    dtc.fit(learning_features, learning_targets)
    prediction = dtc.predict(holdback_features)

    scores = Scoring(holdback_targets, prediction)
    scores.print()

    print()
    print("Test on ML TOPICS")

    ml_topic_df = list()

    for mltopic in ml_dfs:
        print(f"{mltopic}:")
        mltopic_df_list = [ml_dfs[mltopic]]

        ml_topic_df.append(ml_dfs[mltopic])

        ml_fv, ml_tv = lowess_trends_and_features(mltopic_df_list)

        ml_fv = np.array(ml_fv)
        ml_tv = np.array(ml_tv)

        ml_predictions = dtc.predict(ml_fv)

        print(f"Predictions: {ml_predictions}")
        print(f"Real Label: {ml_tv}")
        print()

    print()
    ml_fv, ml_tv = lowess_trends_and_features(ml_topic_df)
    ml_predictions = dtc.predict(ml_fv)
    scores = Scoring(ml_tv, ml_predictions)
    scores.print()

    dot_data = StringIO()
    export_graphviz(dtc, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    graph.write_png('/home/norpheo/Documents/dt_depth3_newmodel.png')
