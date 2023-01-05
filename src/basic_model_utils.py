import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import feature_extraction
from sklearn import model_selection
from sklearn import preprocessing
import xgboost


def get_predefined_split(train_len, dev_len):
    train_idxs = np.full((train_len,), -1, dtype=int)
    dev_idxs = np.zeros((dev_len), dtype=int)
    test_fold = np.append(train_idxs, dev_idxs)
    predefined_split = model_selection.PredefinedSplit(test_fold)
    
    return predefined_split


def create_dict_vector_and_label_features(
    feature_name, split_to_feature_df, label_name='label'):

    label_encoder = preprocessing.LabelEncoder()
    dict_vectorizer = feature_extraction.DictVectorizer()

    x = {}
    y = {}

    x['train'] = dict_vectorizer.fit_transform(
        split_to_feature_df['train'][feature_name])
    y['train'] = label_encoder.fit_transform(
        split_to_feature_df['train'][label_name])

    x['dev'] = dict_vectorizer.transform(
        split_to_feature_df['dev'][feature_name])
    y['dev'] = label_encoder.transform(split_to_feature_df['dev'][label_name])

    x['test'] = dict_vectorizer.transform(
        split_to_feature_df['test'][feature_name])
    y['test'] = label_encoder.transform(split_to_feature_df['test'][label_name])

    return x, y, dict_vectorizer.vocabulary_


def train_xgboost(
    x_train, y_train, x_dev, y_dev, param_grid,
    objective='multi:softmax', scoring='accuracy', eval_metric='mlogloss'):

    # x_train and x_dev are sparse matrices.
    train_len = x_train.shape[0]
    dev_len = x_dev.shape[0]

    predefined_split = get_predefined_split(train_len, dev_len)

    x_train = sparse.vstack([x_train, x_dev])
    y_train = np.append(y_train, y_dev)

    xgb = xgboost.XGBClassifier(
        objective=objective, nthread=1, use_label_encoder=False,
        eval_metric=eval_metric)
    
    grid_search = model_selection.GridSearchCV(
        estimator=xgb, param_grid=param_grid,
        scoring=scoring, n_jobs=-1, cv=predefined_split)
    
    grid_search.fit(x_train, y_train)
    best_xgb = grid_search.best_estimator_
    best_score = grid_search.best_score_

    return best_xgb, best_score


def get_few_shot_sample_indices(labels, sample_percent, random_seed=42):
    df = pd.DataFrame(labels, columns=['label'])
    selected_indices = []
    unique_labels = df['label'].unique()
    sample_percent_per_class = sample_percent / len(unique_labels)
    num_samples_per_class = int(len(labels) * sample_percent_per_class)
    for unique_label in unique_labels:
        class_df = df[df['label'] == unique_label]
        assert len(class_df) >= num_samples_per_class
        indices = class_df.sample(
            n=num_samples_per_class, random_state=random_seed).index.tolist()
        selected_indices.extend(indices)

    total_indices = set(range(len(labels)))
    unselected_indices = list(total_indices - set(selected_indices))

    assert (len(unselected_indices)
               + len(selected_indices)) == len(total_indices)
    return selected_indices, unselected_indices


def get_replace_label_with_pseudo_label_fn(pseudo_labels):
    def replace_fn(example, index):
        example['labels'] = pseudo_labels[index]
        return example
    return replace_fn


def get_replace_label_with_ignored_label_fn(
    ignore_example_indices, ignore_index=-100):
    ignore_example_indices = set(ignore_example_indices)
    def replace_fn(example, index):
        if index in ignore_example_indices:
            example['labels'] = ignore_index
        return example
    return replace_fn