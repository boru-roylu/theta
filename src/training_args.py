from typing import Optional, List
import logging
from dataclasses import dataclass, field

import transformers as tfs


logger = logging.getLogger(__name__)


@dataclass
@tfs.add_start_docstrings(tfs.TrainingArguments.__doc__)
class TrainingArguments(tfs.TrainingArguments):
    """
    compute_metrics_on_device (:obj:`bool`, `optional`, 
        defaults to :obj:`False`):
        Whether to compute metrics on each device and gather the values of
        metrics later. It will save GPU-memory when computing the accuracy of 
        MLM.
    """

    debug_mode: bool = field(
        default=False,
        metadata={'help': ''})

    compute_metrics_on_device: bool = field(
        default=False,
        metadata={'help': ('Whether to compute metrics on each device and gather'
                           'the values of metrics later. It will save GPU-memory'
                           'when computing the accuracy of MLM.')
                 })

    data_name: str = field(
        default=None,
        metadata={'help': 'Dataset name.'})

    task_name: str = field(
        default=None,
        metadata={'help': 'Task name.'})

    model_path: str = field(
        default=None,
        metadata={'help': 'Model path.'})

    model_name: str = field(
        default=None,
        metadata={'help': 'Model name.'})

    model_path: str = field(
        default=None,
        metadata={'help': 'Model path.'})

    overwrite_cached_dataset: bool = field(
        default=False,
        metadata={'help': 'Overwrite cached dataset.'})

    eval_test_on_best_dev_at_end: bool = field(
        default=False,
        metadata={'help': ('Evaluate test set on the model with best dev set '
                           'performance at the end of training')})

    layerwise_learning_rate_decay: float = field(
        default=1.0,
        metadata={'help': ('If layerwise_learning_rate_decay < 1.0, layerwise '
                           'learning rate decay will be applied')})

    freeze_bert_model: bool = field(
        default=False,
        metadata={'help': 'Freeze Bert model.'})

    use_ks_model: bool = field(
        default=False,
        metadata={'help': 'Use KS model.'})

    use_rg_model: bool = field(
        default=False,
        metadata={'help': 'Use RG model.'})

    save_ks_model: bool = field(
        default=False,
        metadata={'help': 'Use KS model.'})

    save_rg_model: bool = field(
        default=False,
        metadata={'help': 'Use RG model.'})

    use_different_learning_rate: bool = field(
        default=False,
        metadata={'help': 'Use different learning rates for different modules.'})

    bert_learning_rate: float = field(
        default=0.0,
        metadata={'help': 'Learning rate for backbone Bert module.'})

    copy_to_init_special_token_embeddings: bool = field(
        default=False,
        metadata={
            'help': 
                'When adding new vocab, initializes from existing embeddings.'
        })

    instance_id: str = field(
        default=None,
        metadata={'help': 'Instance ID on EC2'})

    eval_splits: Optional[List[str]] = field(
        default=None,
        metadata={'help': ('Evaluation splits.')})

    few_shot_ratio: float = field(
        default=None,
        metadata={'help': ('Few shot ratio.')})

    skip_ks_eval: bool = field(
        default=None,
        metadata={'help': ('Skip KS evaluation (use when applying all-document-token).')})

    data_config_path: str =field(
        default=None,
        metadata={'help': 'Data and task configuration.'})

    coordinator_config_path: str =field(
        default=None,
        metadata={'help': 'Hibert configuration.'})

    decoder_config_path: str =field(
        default=None,
        metadata={'help': 'Hibert\'s decoder configuration.'})

    hidden_unit_loss_weight: float = field(
        default=0.0,
        metadata={'help': 'Hidden unit loss weight (alhpa).'})

    init_from_hibert: bool = field(
        default=False,
        metadata={'help': 'Initialize from HiBert.'})

    num_states: int = field(
        default=None,
        metadata={'help': 'Maximum number of states in HMM.'})

    num_clusters: int = field(
        default=None,
        metadata={'help': 'Maximum number of clusters in KMeans.'})

    slurm_job_name: str = field(
        default=None,
        metadata={'help': 'Slurm job name for grouping in wandb.'})

    embedding_name: str = field(
        default=None,
        metadata={'help': 'Name of embeddings for clusters.'})