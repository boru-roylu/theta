import hierarchical_models as hm
import transformers as tfs


def theta_cls_hibert_model_init(train_args, tokenizer, data_config):

    coordinator_config = tfs.AutoConfig.from_pretrained(
        train_args.coordinator_config_path,
        use_cache=False,
        num_labels=data_config['config']['num_labels'])

    use_state_sequence_classifier = \
        getattr(coordinator_config, 'use_state_sequence_classifier', False)

    use_cluster_sequence_classifier = \
        getattr(coordinator_config, 'use_cluster_sequence_classifier', False)

    num_clusters = None
    if use_cluster_sequence_classifier:
        num_clusters = train_args.num_clusters * 2
    num_states = None
    if use_state_sequence_classifier:
        num_states = train_args.num_states

    model = hm.HierarchicalBertModelForConversationClassification.from_pretrained(
        train_args.model_path,
        coordinator_config=coordinator_config,
        num_states=num_states,
        num_clusters=num_clusters,
        use_state_sequence_classifier=use_state_sequence_classifier,
        use_cluster_sequence_classifier=use_cluster_sequence_classifier,
        state_sequence_encoder_type='transformer',
        cluster_sequence_encoder_type='transformer',
        num_labels=data_config['config']['num_labels'])

    return model