import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import conf.constants as const
from torchvision import transforms
from torchvision.utils import save_image
from tensorkit import tensor as T
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset
import gensim.downloader

from graph.event_graph_search import DeepEventGraph
import networkx as nx

from graph.rca_algo import deep_rca_rank

from torch_geometric.nn import GATConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 15
batch_size = 16
learning_rate = 1e-3
likelihood_samples = 100
min_log_std = -5
max_log_std = 5

meta_dim = 100


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels=1):
        super(GCN, self).__init__()
        self.trained = False
        self.conv1 = GATConv(num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)
        x = torch.squeeze(x)
        return x


class RuleDict(nn.Module):
    def __init__(self, deep_rule):
        super(RuleDict, self).__init__()
        self.deep_rule_variables = {}
        self.self_rule_variables = {}
        self.start_probability = {}
        self.deep_rule = deep_rule
        self.trained = False
        self.attention_type = 'transformer'
        self.fc = torch.nn.Linear(meta_dim, meta_dim)
        self.gat = GCN(meta_dim, meta_dim, meta_dim)

    def get_causality_attention(self, EG):
        x = [self.deep_rule.get_bert_vector(n, contain_entity=True) for n in EG]
        x = T.from_numpy(np.asarray(x)).to(device)
        eg_index = nx.to_scipy_sparse_matrix(EG, format='coo')
        eg_index = T.from_numpy(np.asarray([eg_index.row, eg_index.col]), dtype=torch.long).to(device)
        res = self.gnn(x, eg_index)

    def forward(self, event0, event1, direction='d'):
        if direction == 's':
            if (event0, event1) not in self.self_rule_variables:
                self.self_rule_variables[(event0, event1)] = torch.tensor(
                    1.0, dtype=torch.float, requires_grad=True, device=device)
            return T.clip(self.self_rule_variables[(event0, event1)], 0.000, 100.000)
        elif direction == 'd' or direction == 'u':
            if (event0, event1) not in self.deep_rule_variables:
                self.deep_rule_variables[(event0, event1)] = torch.tensor(
                    1.0, dtype=torch.float, requires_grad=True, device=device)
            return T.clip(self.deep_rule_variables[(event0, event1)], 0.000, 100.000)
        else:
            return 2.0

    def attention(self, a, b):
        # ['dot', 'add', 'concat', 'scale_dot', 'bilinear', 'transformer']
        if self.attention_type in ['dot', 'bilinear']:
            return (a * b).sum(dim=-1, keepdim=True)  # [edge_num, heads]
        if self.attention_type == 'add':
            return (self.attention_vector * (a + b).tanh()).sum(dim=-1, keepdim=True)
        if self.attention_type == 'concat':
            return (self.attention_vector * torch.cat([a, b], dim=-1)).sum(dim=-1, keepdim=True)
        if self.attention_type in ['scale_dot', 'transformer']:
            return (a * b).sum(dim=-1, keepdim=True) / self.scale
        return None

    def get_start_probability(self, event):
        if event not in self.start_probability:
            self.start_probability[event] = self.fc(self.deep_rule.get_meta_vector(event))
        return self.start_probability[event]

    def get_parameters(self):
        return list(self.deep_rule_variables.values()) + list(self.self_rule_variables.values()) + list(
            self.start_probability.values())


class DeepRule:
    def __init__(self, catalog, rca_parameters, token_type='bert', model_type='vae'):
        if token_type == 'bert':
            self.rule_space_size = 768
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        elif token_type == 'w2v':
            self.rule_space_size = 300
            self.w2v_model = gensim.downloader.load('word2vec-google-news-300')
        else:
            raise RuntimeError("No such token type: {}".format(token_type))
        self.catalog = catalog
        self.bert_memory = {}
        self.rule_dict = RuleDict(self).to(device)
        self.token_type = token_type
        self.model_type = model_type
        self.gnn = GCN(self.rule_space_size, 100).to(device)
        self.rca_parameters = rca_parameters

    def mini_deep_groot(self, data, require_res=False, require_eg=False):
        pool_events = data['events']['poolEvents']
        domain_events = data['events']['domainEvents']
        pool_list, subG_obj = data['subGraphPools'], data['subGraph']
        for node in subG_obj['nodes']:
            node['events'] = set(node['events'])
        subG = nx.json_graph.node_link_graph(subG_obj, True, True)
        domain = None
        if self.catalog == const.DOMAIN:
            domain = data['domain']
        else:  # cs_md Case
            pool_list = data[const.KEY_GENERAL].split(",")
        eg = DeepEventGraph(pool_events, subG, pool_list, self.catalog)
        # Step 4 Build Event Causal Graph
        eg.add_events()

        # Step 5 Calculate RCA recommendations
        rc_events = eg.get_rc_events()
        print("rc_events")
        print(rc_events)

        EG = eg.event_focus_graph()
        # if len(rc_events) == 0:
        #     EG = eg.event_focus_graph()
        # elif len(rc_events) == 1:
        #     EG = eg.event_focus_graph()
        #     tmp = set()
        #     for e in rc_events:
        #         tmp.update(nx.node_connected_component(EG.to_undirected(), e))
        #     rc_events = list(tmp)
        #     EG = EG.subgraph(rc_events)
        # else:
        #     EG = eg.event_focus_graph()
        #     tmp = set()
        #     for e in rc_events:
        #         tmp.update(nx.node_connected_component(EG.to_undirected(), e))
        #     rc_events = list(tmp)
        #     EG = EG.subgraph(rc_events)
        print("RCA:")
        print(data['RCAEvent'], data['RCAPool'])
        if (data['RCAEvent'], data['RCAPool']) not in EG:
            return None
        if require_eg:
            return EG
        for n in EG:
            self.predict_start_probability(n)
        for u, v, d in EG.edges(data='direction'):
            self.predict_likelihood(u, v, direction=d)
        if require_res:
            return deep_rca_rank(EG, self, limited_start_set=eg.events_to_search, **self.rca_parameters)[
                (data['RCAEvent'], data['RCAPool'])]
        return True

    def train(self, training_data, load_model_path=None, fine_tune=False):
        if load_model_path is not None and os.path.exists(load_model_path):
            self.rule_dict = torch.load(os.path.join(load_model_path, 'rule_dict'))
            if not fine_tune:
                return

        cat_num_epoch = 1
        batch_size = 128

        for i, data in enumerate(training_data):
            print(f"{i} / {len(training_data)} in generating entity data")
            res = self.mini_deep_groot(data)

        print("self.rule_dict.get_parameters()")
        print(self.rule_dict.get_parameters())
        optimizer = torch.optim.Adam(self.rule_dict.get_parameters(), lr=learning_rate * 10)
        loss = 0
        batch_num = 0
        for epoch in range(cat_num_epoch):
            step = 0
            random_index = np.arange(0, len(training_data))
            np.random.shuffle(random_index)
            for i in range(len(training_data)):
                data = training_data[random_index[i]]
                res = self.mini_deep_groot(data, require_res=True)
                if res is not None:
                    loss = loss - res
                    batch_num = batch_num + 1
                    if batch_num >= batch_size:
                        step = step + 1
                        if isinstance(loss, torch.Tensor):
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            print("Epoch[{}/{}], Step: {}, Loss: {:.4f}"
                                  .format(epoch + 1, cat_num_epoch, step, loss.item()))
                        loss = 0
                        batch_num = 0
        self.rule_dict.trained = True
        torch.save(self.rule_dict, 'rule_dict')

    def train_gnn(self, training_data, load_model_path=None, fine_tune=False):
        if load_model_path is not None and os.path.exists(load_model_path):
            self.rule_dict = torch.load(os.path.join(load_model_path, 'gnn'))
            if not fine_tune:
                return

        cat_num_epoch = num_epochs * 2 if len(training_data) < 100 else num_epochs
        print("self.gnn.parameters()")
        print(self.gnn.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=learning_rate)
        loss = 0
        batch_num = 0
        self.gnn.train()
        for epoch in range(cat_num_epoch):
            step = 0
            random_index = np.arange(0, len(training_data))
            np.random.shuffle(random_index)
            for i in range(len(training_data)):
                data = training_data[random_index[i]]
                eg = self.mini_deep_groot(data, require_eg=True)
                if eg is not None:
                    label = 0
                    for i, n in enumerate(eg):
                        if n == (data['RCAEvent'], data['RCAPool']):
                            label = i
                    x = [self.get_bert_vector(n, contain_entity=True) for n in eg]
                    x = T.from_numpy(np.asarray(x)).to(device)
                    eg_index = nx.to_scipy_sparse_matrix(eg, format='coo')
                    eg_index = T.from_numpy(np.asarray([eg_index.row, eg_index.col]), dtype=torch.long).to(device)
                    res = self.gnn(x, eg_index)
                    label = np.expand_dims(label, 0)
                    while len(res.shape) < 2:
                        res = T.expand_dim(res, axis=0)
                    loss = loss + criterion(res, T.from_numpy(label).to(device))
                    batch_num = batch_num + 1
                    if batch_num >= batch_size:
                        step = step + 1
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print("Epoch[{}/{}], Step: {}, Loss: {:.4f}"
                              .format(epoch + 1, cat_num_epoch, step, loss.item()))
                        loss = 0
                        batch_num = 0
        self.gnn.trained = True
        self.gnn.eval()
        # torch.save(self.gnn, 'gnn')

    def predict_rca_by_gnn(self, EG):
        x = [self.get_bert_vector(n, contain_entity=True) for n in EG]
        x = T.from_numpy(np.asarray(x)).to(device)
        eg_index = nx.to_scipy_sparse_matrix(EG, format='coo')
        eg_index = T.from_numpy(np.asarray([eg_index.row, eg_index.col]), dtype=torch.long).to(device)
        res = self.gnn(x, eg_index)
        res = torch.softmax(res, dim=0)
        while len(res.shape) < 1:
            res = T.expand_dim(res, axis=0)
        res_dict = {}
        for i, n in enumerate(EG):
            res_dict[n] = res[i].item()
        return res_dict

    def standlize_name(self, n):
        if n == 'DEPLOYSOFTWARE':
            return 'deploy software'
        if n == 'REFRESHCONFIG':
            return 'refresh config'
        if n == 'DBMarkdown':
            return 'DB markdown'
        if n[:2] == 'r1':
            n = n[2:]
        n = n.replace('_', ' ')
        n = n.replace('-', ' ')
        new_name = ''
        for word in n:
            if word.isupper:
                new_name = new_name.strip() + ' ' + str.lower(word)
            else:
                new_name = new_name + word
        return new_name.strip()

    def get_bert_vector(self, s, contain_entity=False):
        if isinstance(s, tuple):
            event = s[0]
            pool = s[1]
        elif isinstance(s, str):
            if ',' in s:
                event, pool = s.split(',')[0], s.split(',')[1]
                event = event.strip('(').strip(')')
                pool = pool.strip('(').strip(')')
            else:
                event = s
                contain_entity = False
        else:
            raise RuntimeError(f"{s} is not string or tuple")

        if contain_entity:
            s = self.standlize_name(event) + ' in ' + self.standlize_name(pool)
        else:
            s = self.standlize_name(event)
        if s in self.bert_memory:
            return self.bert_memory[s]
        if self.token_type == 'bert':
            inputs = self.bert_tokenizer(s, return_tensors="pt")
            outputs = self.bert_model(**inputs)
            res = T.to_numpy(outputs.last_hidden_state[:, 0], force_copy=True)
            res = np.reshape(res, (-1,))
        elif self.token_type == 'w2v':
            vectors = [self.w2v_model[k] for k in s.split()]
            res = np.mean(np.stack(vectors, axis=0), axis=0)
        else:
            raise RuntimeError(f"{self.token_type} is not supported model")

        self.bert_memory[s] = res
        return res

    def predict_likelihood(self, event0, event1, direction='d'):
        return self.rule_dict(event0[0], event1[0], direction)

    def predict_start_probability(self, event):
        return self.rule_dict.get_start_probability(event[0])

    def predict_entity_likelihood(self, x):
        return 1.0
