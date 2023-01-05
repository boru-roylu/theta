# Unsupervised Learning of Hierarchical Conversation Structure <img src="plot/uwnlp_logo.png" width="8%"> 

<img src="plot/pytorch_logo.png" width="15%"> <img src="plot/huggingface_logo.png" width="15%">

The offical PyTorch implementation of Three-stream Hierarchical Transformer (THETA). Please refer to our paper for details.
**Unsupervised Learning of Hierarchical Conversation Structure**. [Bo-Ru Lu](https://nlp.borulu.com/), Yushi Hu, Hao Cheng, Noah A. Smith, Mari Ostendorf. EMNLP 2022 Findings.
[[paper]](https://arxiv.org/abs/2205.12244)

This code has been written using PyTorch >= 1.13 and HuggingFace >= 4.21.2. If you use any source codes included in this repository in your work, please cite the following paper. The bibtex is listed below:

```text
@inproceedings{lu-etal-2022-unsupervised,
    title = "Unsupervised Learning of Hierarchical Conversation Structure",
    author = "Lu, Bo-Ru and Hu, Yushi and Cheng, Hao and Smith, Noah A. and Ostendorf, Mari",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## Abstract
Human conversations can evolve in many different ways, creating challenges for automatic understanding and summarization. Goal-oriented conversations often have meaningful sub-dialogue structure, but it can be highly domain-dependent. This work introduces an unsupervised approach to learning hierarchical conversation structure, including turn and sub-dialogue segment labels, corresponding roughly to dialogue acts and sub-tasks, respectively. The decoded structure is shown to be useful in enhancing neural models of language for three conversation-level understanding tasks. Further, the learned finite-state sub-dialogue network is made interpretable through automatic summarization.

## Model Architecture
<p align="center">
<img src="plot/model.jpg" width="75%" />
</p>
Overview of THETA conversation encoding. The text of each utterance text is encoded by BERT, and a 1-layer transformer further contextualizes utterance embeddings to generate the text vector U. For structure,
utterances are mapped to K-means dialogue acts (DAs), which are input to an HMM to decode sub-dialogue states. 1-layer transformers are applied to sequences of DAs and sub-dialogue states, yielding cluster vector C and state vector S. The concatenation of U, C and S is fed into a linear layer to obtain the structure-enhanced vector for the predictive task. For simplicity, Emb. and Trans. stand for embedding and transformer, respectively.

## Setup

#### Paths

Create an hidden file named `.env` in the main directory and set the 2 paths as follows.
```bash
# the path to store data, model, and experiment results.
NFS_PARENT_DIR=[set your path]
# the path to store tmp files.
SCRATCH_PARENT_DIR=[set your path]
```

#### Dependency

Follow the [instruction](setup.md) to setup the Conda environment.

For Jupyter Notebook, we use [Visual Studio Code](https://code.visualstudio.com/) extensions to run notebooks on remote GPU machines.
- [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Download Raw Data (Optional)

#### Action-Based Conversations Dataset (ABCD)
Download the Action-Based Conversations Dataset (ABCD) dataset from the [official repository](https://github.com/asappresearch/abcd) ([Chen et al., 2021](https://arxiv.org/abs/2104.00783)).

```bash
wget https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz
``` 

#### CraigslistBargains
Download the preprocessed pickle file ([strategy_vector_data_FULL_Yiheng.pkl](https://drive.google.com/open?id=1WxCvZ__ulT--VRL1oijVCa7tTj8SRydx&authuser=0&usp=drive_link)) of CraigslistBargains dataset from Google drive provided in [DialoGraph repository](https://github.com/rishabhjoshi/DialoGraph_ICLR21) ([Joshi et al. 2021](https://arxiv.org/abs/2106.00920)).

## Download Preprocessed Data & Trained Models
We also provide the preprocessed data, structure, and trained models in [Goolge Drive](https://drive.google.com/drive/folders/18tkllW7wKx07xkGExElUvffuXBijSs7w?usp=share_link).
Please put all directories under the path `$NFS_PARENT_DIR` you defined in the hidden file `.env`.

```bash
# NFS_PARENT_DIR will look like this.
.
├── assignments
├── exp
├── pretrained_models
└── state_assignments
```
Note: the directory `assignments` is for clusters.

## Prepare Structure

#### Domain-adaptive Pretraining (DAPT)
The pretrained models can be downloaded in Goolge Drive. [[Check here]](https://github.com/boru-roylu/THETA2/tree/theta-v1#download-preprocessed-data--trained-models)

```bash
# ABCD
# Trained model will be stored in 
# $NFS_PARENT_DIR/exp/abcd/pretrain/bert_init_from_bert-base-uncased_job_name-bash/seed-42/checkpoint-5000/fp32
# Move the trained model to $NFS_PARENT_DIR/pretrained_models/abcd/bert-base-uncased-abcd-wwm
bash pretrain/abcd/bert.sh
```

```bash
# Craigslist
# Trained model will be stored in 
# $NFS_PARENT_DIR/exp/craigslist/pretrain/bert_init_from_bert-base-uncased_job_name-bash/seed-42/checkpoint-5000/fp32
# Move the trained model to $NFS_PARENT_DIR/pretrained_models/craigslist/bert-base-uncased-craigslist-wwm
bash pretrain/craigslist/bert.sh
```

#### (Cluster) Extract Embeddings and Cluster Embeddings
Create HuggingFace dataset objects.
```bash
bash data_scripts/abcd.sh --task_name pretrain
bash data_scripts/craigslist.sh --task_name pretrain
```

Please refer to the Jupyter Notebooks to create cluster assignment.
- `prepare_structure/cluster/prepare_cluster_abcd.ipynb`
- `prepare_structure/cluster/prepare_cluster_craigslist.ipynb`

The cluster assignments can be downloaded in Goolge Drive. [[Check here]](https://github.com/boru-roylu/THETA2/tree/theta-v1#download-preprocessed-data--trained-models)

#### (State) Learn HMM Topologies via State-splitting
Finishing clustering step is required before proceeding this step.

Create HuggingFace dataset objects (Skip if you already done during clustering).
```bash
bash data_scripts/abcd.sh --task_name pretrain
bash data_scripts/craigslist.sh --task_name pretrain
```

Please refer to the Jupyter Notebooks to create state assignments and learn and plot topologies.
- `prepare_structure/state/prepare_state_abcd.ipynb`
- `prepare_structure/state/prepare_state_craigslist.ipynb`

The state assignments can also be downloaded in Goolge Drive. [[Check here]](https://github.com/boru-roylu/THETA2/tree/theta-v1#download-preprocessed-data--trained-models)

## Train THETA

The checkpoints can be downloaded in Goolge Drive. [[Check here]](https://github.com/boru-roylu/THETA2/tree/theta-v1#download-preprocessed-data--trained-models) 

```bash
# ABCD
bash train_scripts/finetune/abcd/cls-cluster-state-structure-hibert.sh \
    --embedding_name bert_mean_pooler_output \
    --num_clusters 60 --num_states 12
```

```bash
# CraigslistBargains
bash train_scripts/finetune/craigslist/cls-cluster-state-structure-hibert.sh \
    --embedding_name bert_mean_pooler_output \
    --num_clusters 14 --num_states 8
```

## Evaluate THETA

Please refer to the Jupyter Notebooks.

- `eval_scripts/eval_abcd.ipynb`
- `eval_scripts/eval_craigslist.ipynb`

## Bug Report
Feel free to create an issue or send email to roylu@uw.edu.

## License
```
copyright 2022-present https://nlp.borulu.com/

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```
