import os
import tempfile

import torch
import transformers as tfs

import comm
import wandb
import zero_to_fp32


def model_base_name(model_name):
    return os.path.basename(model_name)


def extract_sub_model(ckpt, sub_model_name):
    ckpt = {
        k.replace(f'{sub_model_name}.', ''): v
            for k, v in ckpt.items() if k.startswith(sub_model_name)
    }
    return ckpt


def log_url(output_dir):
    if wandb.run is None:
        return
    url = wandb.run.get_url()
    if url is None:
        return
    url_path = os.path.join(output_dir, 'url.txt')
    with open(url_path, 'w') as f:
        print(url, file=f)


def save_zero3_model(
    zero3_checkpoint_dir, tokenizer=None, config=None):

    output_dir = os.path.join(zero3_checkpoint_dir, 'fp32')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'pytorch_model.bin')

    zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(
        zero3_checkpoint_dir, model_path)

    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    log_url(output_dir)


def save_zero3_sub_model(zero3_checkpoint_dir,
                         save_ks=False,
                         save_gh=False,
                         save_rg=False,
                         ks_tokenizer=None,
                         rg_tokenizer=None,
                         ks_config=None,
                         gh_config=None,
                         rg_config=None):


    with tempfile.NamedTemporaryFile() as temp_file:
        zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(
            zero3_checkpoint_dir, temp_file.name)

        ckpt = torch.load(temp_file.name)
        
        if save_ks:
            output_dir = os.path.join(zero3_checkpoint_dir, 'ks_model')
            os.makedirs(output_dir)
            ks_model = extract_sub_model(ckpt, 'ks_model')
            torch.save(ks_model, os.path.join(output_dir, 'pytorch_model.bin'))
            if ks_tokenizer:
                ks_tokenizer.save_pretrained(output_dir)
            if ks_config:
                ks_config.save_pretrained(output_dir)
            log_url(output_dir)

        if save_rg:
            output_dir = os.path.join(zero3_checkpoint_dir, 'rg_model')
            os.makedirs(output_dir)
            rg_model = extract_sub_model(ckpt, 'rg_model')
            torch.save(rg_model, os.path.join(output_dir, 'pytorch_model.bin'))
            if rg_tokenizer:
                rg_tokenizer.save_pretrained(output_dir)
            if rg_config:
                rg_config.save_pretrained(output_dir)
            log_url(output_dir)


def copy_to_init_special_token_embeddings(
    embeddings, special_token_map, tokenizer, default_token=None):
    for k, params in embeddings.named_parameters():
        new_params = params.data
        for token, special_token in special_token_map.items():
            if default_token:
                token_idx = tokenizer.convert_tokens_to_ids([default_token])[0]
            else:
                token_idx = tokenizer.convert_tokens_to_ids([token])[0]
            special_token_idx = tokenizer.convert_tokens_to_ids(
                [special_token])[0]
            new_params[special_token_idx] = new_params[token_idx]
        params.data.copy_(params)


def get_extra_log_function(train_args, data_config):
    def extra_log_function():
        if not comm.is_local_master():
            return 

        if train_args.coordinator_config_path:
            coordinator_config = tfs.AutoConfig.from_pretrained(
                train_args.coordinator_config_path,
                use_cache=False,
                num_labels=data_config['config']['num_labels'])
            coordinator_config_dict = {
                f'coordinator_config/{k}': v 
                for k, v in coordinator_config.to_dict().items()
            }
            wandb.config.update(coordinator_config_dict, allow_val_change=True)

        encoder_config = tfs.AutoConfig.from_pretrained(train_args.model_path)
        encoder_config_dict = {
            f'encoder_config/{k}': v 
                for k, v in encoder_config.to_dict().items()
        }
        data_config_dict = {
            f'data_config/{k}': v for k, v in data_config.items()
        }
        wandb.config.update(encoder_config_dict, allow_val_change=True)
        wandb.config.update(data_config_dict, allow_val_change=True)

    return extra_log_function


def inference(ds, raw_ds, split, train_args, our_trainer):
    # To initial comet_ml.Experiment Dummy log for inference mode: 
    # `do_train = False` and `do_eval = True`.
    if not our_trainer.args.do_train and our_trainer.args.do_eval:
        logs = {}
        our_trainer.control = our_trainer.callback_handler.on_log(
            our_trainer.args, our_trainer.state, our_trainer.control, logs)
    tags = [split]
    if comm.is_local_master():
        experiment = comet_ml.get_global_experiment()
        if experiment:
            if train_args.use_ks_model:
                tags.append(train_args.ks_model_name)
            if train_args.use_rg_model:
                tags.append(train_args.rg_model_name)
            experiment.add_tags(tags)
            experiment.set_name(split)

    our_trainer.evaluate(eval_dataset=ds[split],
                         eval_raw_dataset=raw_ds[split],
                         metric_key_prefix=split)