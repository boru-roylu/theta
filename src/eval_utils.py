import torch
from transformers.utils import logging
import sklearn


logger = logging.get_logger(__name__)


def compute_roc_auc(y_trues, y_probs, prob_idx, pos_label):
    fpr, tpr, thresh = sklearn.metrics.roc_curve(
        y_trues, y_probs[:, prob_idx], pos_label=pos_label)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc


def compute_pr_auc(y_trues, y_probs, prob_idx, pos_label):
    p, r, thresh = sklearn.metrics.precision_recall_curve(
        y_trues, y_probs[:, prob_idx], pos_label=pos_label)
    auc = sklearn.metrics.auc(r, p)
    return auc


def get_classification_finetuning_post_process_function(num_labels):
    def post_process_function(
        inputs, others, predictions, metric_key_prefix, global_step):
        logits = predictions['logits']
        labels = inputs['labels']

        num_classes = logits.shape[1]
        assert num_labels == num_classes

        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
        probs = torch.softmax(logits.float(), dim=1)
        predictions = torch.argmax(logits, dim=1)

        results = {}
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        results['accuracy'] = accuracy
    
        if num_classes == 2:
            roc_auc = compute_roc_auc(labels, probs, 0, 0)
            pr_auc = compute_pr_auc(labels, probs, 0, 0)
            results['roc_auc'] = roc_auc
            results['pr_auc'] = pr_auc
    
        macro_f1 = sklearn.metrics.f1_score(labels, predictions, average='macro')
        micro_f1 = sklearn.metrics.f1_score(labels, predictions, average='micro')
        weighted_f1 = sklearn.metrics.f1_score(labels, predictions, average='weighted')
        weighted_precision = sklearn.metrics.precision_score(labels, predictions, average='weighted')
        weighted_recall = sklearn.metrics.recall_score(labels, predictions, average='weighted')
        results['macro_f1'] = macro_f1
        results['micro_f1'] = micro_f1
        results['weighted_f1'] = weighted_f1
        results['weighted_precision'] = weighted_precision
        results['weighted_recall'] = weighted_recall
    
        return results 
    return post_process_function


def get_pretraining_post_process_function(
    tokenizer, mlm_labels_key='mlm_labels'):
    def post_process_function(
        inputs, others, predictions, metric_key_prefix, global_step):

        def masked_mean(labels, preds):
            labels = torch.from_numpy(labels)
            preds = torch.from_numpy(preds)
            mask = labels == -100
            correct = preds == labels
            correct.masked_fill_(mask, 0)
            num_valid_tokens = (~mask).sum()
            if num_valid_tokens == 0:
                return 0
            acc = correct.sum() / num_valid_tokens
            return float(acc)

        result = {}

        mlm_labels = inputs[mlm_labels_key]
        mlm_preds = others['mlm_preds']
        mlm_acc = masked_mean(mlm_labels, mlm_preds)
        result['mlm_accuracy'] = mlm_acc

        return result
    return post_process_function


def clustering_scores(labels, preds):
    scores = {
        'macro_f1': sklearn.metrics.f1_score(labels, preds, average='macro'),
        'micro_f1': sklearn.metrics.f1_score(labels, preds, average='micro'),
        'weighted_f1': \
            sklearn.metrics.f1_score(labels, preds, average='weighted'),
        'v_measure': sklearn.metrics.v_measure_score(labels, preds),
        'adj_rand_idx': sklearn.metrics.adjusted_rand_score(labels, preds),
        'adj_mutual_info': \
            sklearn.metrics.adjusted_mutual_info_score(labels, preds),
        'kappa_score': sklearn.metrics.cohen_kappa_score(labels, preds),
    }
    return scores