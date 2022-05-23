from typing import Union, List
import collections
import copy
import json

import faiss
import graphviz
import numpy as np
import textwrap
import pickle
import pomegranate as pg


class ClusteringBase(object):
    def train(self, inputs):
        raise NotImplementedError
        
    def predict(self, inputs):
        raise NotImplementedError
        
    @staticmethod
    def from_pretrained(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
        
    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    def predict(self, inputs):
        outputs = self.model.predict(inputs)
        return outputs

        
class FaissClustering(ClusteringBase):
    def __init__(self,
                 dim=None,
                 num_clusters=None,
                 use_gpu=False,
                 niter=100,
                 nredo=1,
                 verbose=0,
                 index=None,
                 cuda_device=None,
                 seed=1234):
        
        if index:
            self.dim = index.d
            self.model = None
        else:
            assert dim
            assert num_clusters
            self.dim = dim
            self.num_clusters = num_clusters
            self.model = faiss.Clustering(dim, num_clusters)
            self.model.seed = seed
            self.model.verbose = bool(verbose)
            self.model.niter = niter
            self.model.nredo = nredo
            # otherwise the kmeans implementation sub-samples the training set
            self.model.max_points_per_centroid = 50000000
            index = faiss.IndexFlatL2(dim)

        if use_gpu:
            if cuda_device:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, device=cuda_device, index=index)
                #index = faiss.index_cpu_to_gpu(index)
            else:
                index = faiss.index_cpu_to_all_gpus(index)
        self.index = index

    def centers(self):
        assert self.model
        centroids = faiss.vector_float_to_array(self.model.centroids)
        centroids = centroids.reshape(self.num_clusters, self.dim) 
        return centroids

    def train(self, inputs):
        self.model.train(inputs, self.index)

    def predict(self, inputs):
        distances, assignments = self.index.search(inputs, 1)
        distances = distances.reshape(-1)
        assignments = assignments.reshape(-1)
        return distances, assignments

    @classmethod
    def from_pretrained(cls, model_path, use_gpu=False):
        index = faiss.read_index(model_path)
        return cls(index=index, use_gpu=use_gpu)
        
    def save_model(self, model_path):
        index = faiss.IndexFlatL2(self.dim)
        index.add(self.centers())
        faiss.write_index(index, model_path)

    
def normalize(dic):
    _sum = sum(dic.values())
    return {k: v/_sum for k, v in dic.items()}


def entropy(prob):
    if isinstance(prob, list):
        prob = np.array(prob)
    log_prob = np.log2(np.clip(prob, 1e-12, 1))
    e = -np.sum(prob * log_prob, axis=prob.ndim-1)
    return e


class DiscreteSequenceClustering(ClusteringBase):
    def __init__(self,
                 init_min_seq_len: int,
                 cluster_vocab: List[Union[int, str]],
                 num_init_states: int = 3,
                 self_loop_prob: float = 0.8,
                 head_tail_ratio: float = 0.05,
                 topk_edges: int = 3):
        """Instantiates a DiscreteSequenceClustering object.

        Args:
            init_min_seq_len: Minimum of sequence length to initialize HMM.
            cluster_vocab: Vocabulary of discrete clusters.
            num_init_states: # initial states.
            self_loop_prob: Initial value for self-loop edges. 
            head_tail_ratio: The ratios of head and tail segments in sequences.
            topk_edges: # of top k edges that will be kept in each interation.
        """
        assert num_init_states >= 2, 'At least 2 initial states.'
        self.num_states = num_init_states
        self.num_init_states = num_init_states
        self.init_min_seq_len = init_min_seq_len
        self.cluster_vocab = cluster_vocab
        self.self_loop_prob = self_loop_prob
        self.head_tail_ratio = head_tail_ratio
        self.topk_edges = topk_edges
        self.num_merging = 0
        self.num_temperal_split = 0
        self.num_vertical_split = 0

    @property
    def num_splits(self):
        return self.num_temperal_split + self.num_vertical_split
    
    @property
    def state_index_to_name(self):
        dic = {}
        for i, state in enumerate(
            json.loads(self.model.to_json())['states']):
            dic[i] = state['name']
        return dic

    @property
    def state_index_to_simplified_name(self):
        state_index_to_simplified_name = {}
        for index, name in self.state_index_to_name.items():
            name = name.split(' <- ')[0].strip()
            state_index_to_simplified_name[index] = name
        return state_index_to_simplified_name

    @staticmethod
    def create_model(states, edges, model):
        model.add_states(states)
        for edge in edges:
            model.add_transition(*edge)
        model.bake()
        return model
    
    def build(self, sequneces):
        self.model, self.state_to_parent = self._init_model(sequneces)
        
    def _init_model(self, sequences):
        num_init_states = self.num_init_states
        self_loop_prob = self.self_loop_prob
        head_tail_ratio = self.head_tail_ratio
        init_min_seq_len = self.init_min_seq_len
        
        # Counts the number of clusters for each segment (state).
        # -2 is for head and tail.
        num_middle_states = num_init_states - 2
        segment_ratios = [head_tail_ratio]
        segment_ratios += [(1 - head_tail_ratio * 2) / 
                           (num_middle_states)] * num_middle_states
        segment_ratios = np.array(segment_ratios)
        
        segment_counters = [collections.Counter() for _ in range(num_init_states)]
        for seq in sequences:
            if len(seq) < init_min_seq_len:
                continue
            segment_lens = segment_ratios * len(seq)
            segment_lens = np.ceil(segment_lens)
            segment_lens = segment_lens.astype(int)
            split_points = np.cumsum(segment_lens)
            split_points = np.clip(split_points, 0, len(seq))
            if split_points[-1] == len(seq):
                split_points[-1] = split_points[-1] - 1
            segments = np.split(seq, split_points)
            assert len(segments) == num_init_states
            for i, clusters in enumerate(segments):
                segment_counters[i] += collections.Counter(clusters)
                
        # Creats HMM states.
        states = []
        base_cnt = {c: 0 for c in self.cluster_vocab}
        for i in range(num_init_states):
            cnt = copy.deepcopy(base_cnt)
            cnt.update(segment_counters[i])
            emission_probs = normalize(cnt)
            name = f'S{i:02}'
            distrib = pg.DiscreteDistribution(emission_probs)
            state = pg.State(distrib, name=name)
            states.append(state)

        model = pg.HiddenMarkovModel(name='ConversationalHMM')
        
        # Adds edges between states.
        edges = [(model.start, states[0], 1.0)]
        state_to_parent = dict()
        for i in range(num_init_states):
            
            # Self-loop.
            edge = (states[i], states[i], self_loop_prob)
            edges.append(edge)
            
            if i == num_init_states - 1:
                # Only end state.
                edge_prob = 1 - self_loop_prob
            else:
                # Next states and end state: n + 1 states.
                n = num_init_states - (i + 1)
                edge_prob = (1 - self_loop_prob) / (n + 1)
                for j in range(i, num_init_states-1):
                #for j in range(i, i+1):
                    edge = (states[i], states[j+1], edge_prob)
                    edges.append(edge)
                state_to_parent[states[i+1].name] = states[i].name
                
            edge = (states[i], model.end, edge_prob)
            edges.append(edge)
        state_to_parent['S00'] = model.start.name 
        
        states.extend([model.start, model.end])
        model = DiscreteSequenceClustering.create_model(states, edges, model)
        return model, state_to_parent
    
    @staticmethod
    def get_edges_and_states(model):
        model_json = json.loads(model.to_json())
        
        def get_named_edges():
            idx2name = {i: s['name'] for i, s in enumerate(model_json['states'])} 
            named_edges = []
            for edge in model_json['edges']:
                edge_start = idx2name[edge[0]]
                edge_end = idx2name[edge[1]]
                prob = edge[2]
                named_edges.append([edge_start, edge_end, prob])
            return named_edges
        
        def get_states():
            dummy_states = []
            state_to_emission_probs = collections.OrderedDict()
            for s in model_json['states']:
                name = s['name']
                if 'start' in name or 'end' in name:
                    dummy_states.append(name)
                else:
                    state_to_emission_probs[name] = s['distribution']['parameters'][0]
            return state_to_emission_probs, dummy_states
        
        named_edges = get_named_edges()
        state_to_emission_probs, dummy_states = get_states()
        return named_edges, state_to_emission_probs, dummy_states
    
    @staticmethod
    def get_name_to_state(state_to_emission_probs,
                          split_state_name,
                          new_state_name,
                          model,
                          zero_out_highest_emission_prob=False):
        name_to_state = {}
        for name, ep in state_to_emission_probs.items(): 
            ep = {int(k): v for k, v in ep.items()}
            name_to_state[name] = pg.State(pg.DiscreteDistribution(ep), name=name)
            if name == split_state_name:
                if zero_out_highest_emission_prob:
                    max_key = max(ep, key=ep.get)
                    ep[max_key] = 0.0
                name_to_state[new_state_name] = pg.State(
                    pg.DiscreteDistribution(ep), name=new_state_name)
        name_to_state[model.start.name] = model.start
        name_to_state[model.end.name] = model.end
        return name_to_state
    
    @staticmethod
    def prune_edges(named_edges,
                    model_start_state,
                    model_end_state,
                    topk,
                    min_transition_prob=1e-4):
        state_to_edges = collections.defaultdict(list)
        reserved_named_edges = []

        states = set()
        for start, end, prob in named_edges:
            # Keeps self-loop or end-state-aiming edges.
            if end == model_end_state or start == end:
                reserved_named_edges.append((start, end, prob))
                continue
            state_to_edges[start].append((start, end, prob))
            states.add(start)
            states.add(end)

        state_to_num_coming_edges = collections.defaultdict(int)
        pruned_state_to_edges = collections.defaultdict(list)
        for state, edges in state_to_edges.items():
            # Sorts by prob.
            sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)

            # Keeps topk and large enough edges.
            for edge in sorted_edges[:topk]:
                start, end, prob = edge
                if prob > min_transition_prob:
                    reserved_named_edges.append(edge)
                else:
                    pruned_state_to_edges[end].append(edge)

            # Saves pruned edges for adding them back if needed.
            for edge in sorted_edges[topk:]:
                _, end, prob = edge
                pruned_state_to_edges[end].append(edge)

        # Counts number of coming edges for each state.
        for start, end, prob in reserved_named_edges:
            if end == model_end_state or start == end:
                continue
            state_to_num_coming_edges[end] += 1

        # Adds edges back if there is no coming edge for the state.
        for state in states:
            if state == model_start_state or state == model_end_state:
                continue

            if state_to_num_coming_edges[state] == 0:
                reserved_named_edges.append(pruned_state_to_edges[state][0])

        return reserved_named_edges

    def temperal_split(self,
                       split_state_name,
                       named_edges,
                       state_to_emission_probs):
        """Adds a new state by temperal split.
        
        Args:
            split_state_name: str. The name of the state that is going to 
                be split.
            named_edges: List[Tuple[str, str, float]]. A list of edges.
            state_to_emission_probs: Mapping[str, Mapping[str, float]]. 
                A dictionary of state names to their emission probabilities.
            
        Returns:
            model: pg.HiddenMarkovModel. The model with the new splitting state.
            state_to_parent: dictionary. The mapping from state name to 
                its parent name.
        """
        
        new_state_name = f'T{self.num_temperal_split:02} <- {split_state_name}'
        name_to_state = DiscreteSequenceClustering.get_name_to_state(
            state_to_emission_probs, split_state_name, new_state_name, self.model)
        new_state = name_to_state[new_state_name]
        split_state = name_to_state[split_state_name]
        
        model = pg.HiddenMarkovModel(name='ConversationalHMM')
        name_to_state[model.start.name] = model.start
        name_to_state[model.end.name] = model.end

        state_to_parent = copy.deepcopy(self.state_to_parent)
        parent_name = state_to_parent[split_state_name]
        state_to_parent[new_state_name] = parent_name
        state_to_parent[split_state_name] = new_state_name
    
        # Temperal split.
        new_named_edges = []
        #remaining_prob = 1.0
        for start_name, end_name, prob in named_edges:
            start_state = name_to_state[start_name]
            end_state = name_to_state[end_name]
            
            # s: split state.
            # s': new state.
            #                     prob
            # original states: p ------> s
            # after splitting:    prob       prob
            #                  p ------> s' ------> s
            # Note: we didn't plot self-loop above.

            # Self-loop.
            if start_name == end_name:
                new_named_edges.append((start_state, end_state, prob))
                if start_name == split_state_name:
                    new_named_edges.append((new_state, new_state, prob))
                continue
            
            if end_name == self.model.end.name:
                new_named_edges.append((start_state, end_state, prob))
                if start_name == split_state_name:
                    new_named_edges.append((new_state, end_state, prob / 2))
                    # edge: s' --> s.
                    new_named_edges.append((new_state, split_state, prob / 2))
                continue

            # Takes incoming edges of the split state.
            if end_name == split_state_name:
                new_named_edges.append((start_state, new_state, prob))
                continue

            # Copies outgoing edges of the split state to the new state.
            if start_name == split_state_name:
                new_named_edges.append((split_state, end_state, prob))
                new_named_edges.append((new_state, end_state, prob))
                continue

            new_named_edges.append((start_state, end_state, prob))
                
        #new_named_edges.append((new_state, split_state, remaining_prob))
        new_named_edges = DiscreteSequenceClustering.prune_edges(
            new_named_edges, model.start, model.end, self.topk_edges)
            
        model = DiscreteSequenceClustering.create_model(
            list(name_to_state.values()), new_named_edges, model)
    
        return model, state_to_parent
    
    def vertical_split(self,
                       split_state_name,
                       named_edges,
                       state_to_emission_probs):
        """Adds a new state by vertical split.
        
        Args:
            split_state_name: str. The name of the state that is going to 
                be split.
            named_edges: List[Tuple[str, str, float]]. A list of edges.
            state_to_emission_probs: Mapping[str, Mapping[str, float]]. 
                A dictionary of state names to their emission probabilities.
            
        Returns:
            model: pg.HiddenMarkovModel. The model with the new splitting state.
            new_state_name: str. The name of the new splitting state.
        """
        
        new_state_name = f'V{self.num_vertical_split:02} <- {split_state_name}'
        name_to_state = DiscreteSequenceClustering.get_name_to_state(
            state_to_emission_probs, split_state_name, new_state_name,
            self.model, zero_out_highest_emission_prob=True)
        new_state = name_to_state[new_state_name]
        split_state = name_to_state[split_state_name]
        
        model = pg.HiddenMarkovModel(name='ConversationalHMM')
        name_to_state[model.start.name] = model.start
        name_to_state[model.end.name] = model.end
        
        state_to_parent = copy.deepcopy(self.state_to_parent)
        parent_name = state_to_parent[split_state_name]
        state_to_parent[new_state_name] = parent_name
    
        # Vertical split.
        new_named_edges = []
        for start_name, end_name, prob in named_edges:
            start_state = name_to_state[start_name]
            end_state = name_to_state[end_name]
            
            # Divides the transition prob by 2.
            #                     prob
            # original states: s ------> c
            # 
            # after splitting:                   prob/2
            #                  (split state) s  ------> c 
            #                  (new state)   s' -------/   
            #                                    prob/2
            # Note: we didn't plot self-loop above.

            # Self-loop
            if start_name == end_name:
                new_named_edges.append((start_state, end_state, prob))
                if start_name == split_state_name:
                    new_named_edges.append((new_state, new_state, prob))
                continue

            # Shares incoming edges of the split state with the new state.
            if end_name == split_state_name:
                new_named_edges.append((start_state, split_state, prob / 2))
                new_named_edges.append((start_state, new_state, prob / 2))
                continue

            # Copies outgoing edges of the split state to the new state.
            if start_name == split_state_name:
                new_named_edges.append((split_state, end_state, prob))
                new_named_edges.append((new_state, end_state, prob))
                continue

            new_named_edges.append((start_state, end_state, prob))

        new_named_edges = DiscreteSequenceClustering.prune_edges(
            new_named_edges, model.start, model.end, self.topk_edges)
        model = DiscreteSequenceClustering.create_model(
            list(name_to_state.values()), new_named_edges, model)
    
        return model, state_to_parent
    
    @staticmethod
    def train(model, sequences, max_iterations, n_jobs):
        model.fit(sequences,
                  algorithm='baum-welch',
                  emission_pseudocount=1,
                  stop_threshold=20,
                  max_iterations=max_iterations,
                  verbose=True,
                  n_jobs=n_jobs)
        return model
    
    def split_and_train(self, train_sequences, dev_sequences, max_iterations, n_jobs):
        (named_edges,
         state_to_emission_probs,
         dummy_states) = DiscreteSequenceClustering.get_edges_and_states(self.model)
        
        max_entropy_state_name = max(
            state_to_emission_probs.items(),
            key=lambda x: entropy(list(x[1].values())))[0]
        print(f'The state with max entropy is: {max_entropy_state_name}.')
        
        print('Try temperal split.')
        t_model, t_state_to_parent = self.temperal_split(max_entropy_state_name,
                                                         named_edges,
                                                         state_to_emission_probs)
        t_model = DiscreteSequenceClustering.train(t_model,
                                                   train_sequences,
                                                   max_iterations,
                                                   n_jobs)
        t_log_prob = sum(t_model.log_probability(seq) for seq in dev_sequences) 
        print(f'Temperal log prob = {t_log_prob:.4f}')
        
        print('Try vertical split.')
        v_model, v_state_to_parent = self.vertical_split(max_entropy_state_name,
                                                         named_edges,
                                                         state_to_emission_probs)
        v_model = DiscreteSequenceClustering.train(v_model,
                                                   train_sequences,
                                                   max_iterations,
                                                   n_jobs)
        v_log_prob = sum(v_model.log_probability(seq) for seq in dev_sequences)
        print(f'Vertical log prob = {v_log_prob:.4f}')

        if t_log_prob > v_log_prob:
            print(f'Use temperal split.')
            self.model = t_model
            self.state_to_parent = t_state_to_parent
            self.num_temperal_split += 1
        else:
            print(f'Use vertical split.')
            self.model = v_model
            self.state_to_parent = v_state_to_parent
            self.num_vertical_split += 1
        self.num_states += 1


def plot_hmm(hmm, topk_clusters, cluster_to_utterances=None, title=None):
    min_edge_prob = 1e-3
    model = hmm.model
    (named_edges,
     state_name_to_emission_probs,
     dummy_states) = DiscreteSequenceClustering.get_edges_and_states(model)
    
    g = graphviz.Digraph('G', engine='dot')
    c0 = graphviz.Digraph('cluster_0')
    c0.body.append('style=filled')
    c0.body.append('color=white')
    c0.attr('node')
    c0.node_attr.update(color='#D0F0FB', style='filled', fontsize="14", shape='box')
    c0.edge_attr.update(color='white')

    state_name_to_incoming_edges = collections.defaultdict(list)
    for start_name, end_name, prob in named_edges:
        # Excludes self-loop edge.
        if start_name == end_name:
            continue
        state_name_to_incoming_edges[end_name].append((start_name, end_name, prob))

    # Emission probs.
    text_width = 70
    state_name_to_node_attrs = collections.OrderedDict()
    sn2ep = list(state_name_to_emission_probs.items())
    state_name_to_emission_prob = list(state_name_to_emission_probs.items())
    for name, ep in state_name_to_emission_prob:
        incoming_edges = state_name_to_incoming_edges[name]
        if len(incoming_edges) == 1 and incoming_edges[0][2] < min_edge_prob:
            color = '#d3d3d3' 
        else:
            color = '#D0F0FB'
        topk = sorted(
            ep.items(), key=lambda x: x[1], reverse=True)[:topk_clusters]
        string = name + "\n"
        for cluster, prob in topk:
            if cluster_to_utterances:
                utterances = cluster_to_utterances[cluster]
                utterances = [textwrap.fill(utt, text_width).replace('\n', '-\l')
                              for utt in utterances]
                utterances = "\l\l".join(utterances)
            else:
                utterances = ''
            string += f'cluster = {cluster}; prob = {prob:.3f}\n'
            string += utterances
            string += '\l'
            string += '-' * 80
            string += '\n'
        state_name_to_node_attrs[name] = {
            'name': name,
            'label': string,
            'color': color,
        }

    # Model start and end states.
    for s in dummy_states:
        c0.node(s)
        
    # Normal states.
    for name in state_name_to_emission_probs.keys():
        attrs = state_name_to_node_attrs[name] 
        #c0.node(name, state_string)
        c0.node(**attrs)
        
    for start_name, end_name, prob in named_edges:
        if prob < min_edge_prob:
            label = f'{prob:e}'
            color = '#808080'
        else:
            label = f'{prob:.3f}'
            color = '#005FFE'
        if end_name == model.end.name:
            c0.edge(start_name, end_name, label=label, len='3.00', style='dashed', color=color, fontcolor=color)
        else:
            c0.edge(start_name, end_name, label=label, len='3.00', color=color, fontcolor=color)
        
    g.subgraph(c0)
    g.edge_attr.update(arrowsize='1')
    g.body.append('rankdir=LR\n')
    g.body.append('size="16,10"\n')
    if not title:
        title = f'num_splits_{hmm.num_splits}_num_states_{hmm.num_states}_num_merging_{hmm.num_merging}'
    g.body.append('labelloc="t"\n')
    g.body.append(f'label="{title}"\n')
    return g