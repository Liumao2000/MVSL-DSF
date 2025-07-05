# Network.py
import torch
import torch.nn.functional as F
from torch import nn
import dgl.function as fn
import dgl
from dgl.nn.pytorch import edge_softmax
from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from functools import partial
import math

Dropout = 0.3

class FingerprintCNN(nn.Module):
    def __init__(self, fp_dim=2048, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, hidden_dim)

    def forward(self, fp):
        fp = fp.unsqueeze(1)
        features = self.conv(fp).squeeze(-1)
        return self.fc(features)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size
        self.all_head_size = hidden_size * num_attention_heads

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores_1(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_2(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, int(self.attention_head_size / self.num_attention_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores_1(mixed_query_layer)
        key_layer = self.transpose_for_scores_1(mixed_key_layer)
        value_layer = self.transpose_for_scores_1(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_head_size * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size * num_attention_heads, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class multimodal_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(multimodal_Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, num_attention_heads, hidden_dropout_prob)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class GlobalPool(nn.Module):
    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()
        self.compute_logits_1 = nn.Sequential(nn.Linear(2 * feat_size, 1), nn.LeakyReLU())
        self.compute_logits_2 = nn.Sequential(nn.Linear(2 * feat_size, 1), nn.LeakyReLU())
        self.compute_logits_3 = nn.Sequential(nn.Linear(2 * feat_size, 1), nn.LeakyReLU())

        self.project_nodes_1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_size, feat_size))
        self.project_nodes_2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_size, feat_size))
        self.project_nodes_3 = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_size, feat_size))

        self.multi_to_one_node = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_size * 3, feat_size))

        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        with g.local_scope():
            g.ndata['z_1'] = self.compute_logits_1(torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['z_2'] = self.compute_logits_2(torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['z_3'] = self.compute_logits_3(torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))

            g.ndata['a_1'] = dgl.softmax_nodes(g, 'z_1')
            g.ndata['a_2'] = dgl.softmax_nodes(g, 'z_2')
            g.ndata['a_3'] = dgl.softmax_nodes(g, 'z_3')

            g.ndata['hv_1'] = self.project_nodes_1(node_feats)
            g.ndata['hv_2'] = self.project_nodes_2(node_feats)
            g.ndata['hv_3'] = self.project_nodes_3(node_feats)

            g_repr_1 = dgl.sum_nodes(g, 'hv_1', 'a_1')
            g_repr_2 = dgl.sum_nodes(g, 'hv_2', 'a_2')
            g_repr_3 = dgl.sum_nodes(g, 'hv_3', 'a_3')

            context_1 = F.elu(g_repr_1)
            context_2 = F.elu(g_repr_2)
            context_3 = F.elu(g_repr_3)

            context = F.elu(self.multi_to_one_node(torch.cat([context_1, context_2, context_3], dim=1)))

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a1']
            else:
                return self.gru(context, g_feats)

class AttentiveFPReadout(nn.Module):
    def __init__(self, feat_size, num_timesteps=2, dropout=Dropout):
        super(AttentiveFPReadout, self).__init__()
        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')
        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats

class AttentiveGRU1(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()
        self.edge_transform_1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(edge_feat_size, edge_hidden_size))
        self.edge_transform_2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(edge_feat_size, edge_hidden_size))
        self.edge_transform_3 = nn.Sequential(nn.Dropout(dropout), nn.Linear(edge_feat_size, edge_hidden_size))
        self.multi_to_cat_attention = nn.Sequential(nn.Dropout(dropout), nn.Linear(edge_hidden_size * 3, edge_hidden_size))
        self.multi_to_cat_node = nn.Sequential(nn.Dropout(dropout), nn.Linear(edge_hidden_size * 3, edge_hidden_size))
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.edge_transform_1[1].reset_parameters()
        self.edge_transform_2[1].reset_parameters()
        self.edge_transform_3[1].reset_parameters()
        self.multi_to_cat[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e1'] = edge_softmax(g, edge_logits[0]) * self.edge_transform_1(edge_feats[0])
        g.update_all(fn.copy_e('e1', 'm'), fn.sum('m', 'c'))
        context_1 = F.elu(g.ndata['c'])

        g.edata['e2'] = edge_softmax(g, edge_logits[1]) * self.edge_transform_2(edge_feats[1])
        g.update_all(fn.copy_e('e2', 'm'), fn.sum('m', 'c'))
        context_2 = F.elu(g.ndata['c'])

        g.edata['e3'] = edge_softmax(g, edge_logits[2]) * self.edge_transform_3(edge_feats[2])
        g.update_all(fn.copy_e('e3', 'm'), fn.sum('m', 'c'))
        context_3 = F.elu(g.ndata['c'])

        context = torch.cat([context_1, context_2, context_3], dim=1)
        context = self.multi_to_cat_attention(context)

        node_feats = torch.cat([node_feats[0], node_feats[1], node_feats[2]], dim=1)
        node_feats = self.multi_to_cat_node(node_feats)
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()
        self.project_node = nn.Sequential(nn.Dropout(dropout), nn.Linear(node_feat_size, edge_hidden_size))
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()
        self.project_node1_1 = nn.Sequential(nn.Linear(node_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_node1_2 = nn.Sequential(nn.Linear(node_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_node1_3 = nn.Sequential(nn.Linear(node_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_edge1_1 = nn.Sequential(nn.Linear(node_feat_size + edge_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_edge1_2 = nn.Sequential(nn.Linear(node_feat_size + edge_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_edge1_3 = nn.Sequential(nn.Linear(node_feat_size + edge_feat_size, graph_feat_size), nn.LeakyReLU())
        self.project_edge2_1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * graph_feat_size, 1), nn.LeakyReLU())
        self.project_edge2_2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * graph_feat_size, 1), nn.LeakyReLU())
        self.project_edge2_3 = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * graph_feat_size, 1), nn.LeakyReLU())
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        self.project_node1_1[0].reset_parameters()
        self.project_node1_2[0].reset_parameters()
        self.project_node1_3[0].reset_parameters()
        self.project_edge1_1[0].reset_parameters()
        self.project_edge1_2[0].reset_parameters()
        self.project_edge1_3[0].reset_parameters()
        self.project_edge2_1[1].reset_parameters()
        self.project_edge2_2[1].reset_parameters()
        self.project_edge2_3[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2_1(self, edges):
        return {'he2_1': torch.cat([edges.dst['hv_new_1'], edges.data['he1_1']], dim=1)}

    def apply_edges2_2(self, edges):
        return {'he2_2': torch.cat([edges.dst['hv_new_2'], edges.data['he1_2']], dim=1)}

    def apply_edges2_3(self, edges):
        return {'he2_3': torch.cat([edges.dst['hv_new_3'], edges.data['he1_3']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        logits = []
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new_1'] = self.project_node1_1(node_feats)
        g.ndata['hv_new_2'] = self.project_node1_2(node_feats)
        g.ndata['hv_new_3'] = self.project_node1_3(node_feats)
        g.edata['he'] = edge_feats
        g.apply_edges(self.apply_edges1)

        g.edata['he1_1'] = self.project_edge1_1(g.edata['he1'])
        g.edata['he1_2'] = self.project_edge1_2(g.edata['he1'])
        g.edata['he1_3'] = self.project_edge1_3(g.edata['he1'])

        multi_node_feats = [g.ndata['hv_new_1'], g.ndata['hv_new_2'], g.ndata['hv_new_3']]

        g.apply_edges(self.apply_edges2_1)
        g.apply_edges(self.apply_edges2_2)
        g.apply_edges(self.apply_edges2_3)

        logits1 = self.project_edge2_1(g.edata['he2_1'])
        logits2 = self.project_edge2_2(g.edata['he2_2'])
        logits3 = self.project_edge2_3(g.edata['he2_3'])

        multi_edge_feats = [g.edata['he1_1'], g.edata['he1_2'], g.edata['he1_3']]

        logits.append(logits1)
        logits.append(logits2)
        logits.append(logits3)
        return self.attentive_gru(g, logits, multi_edge_feats, multi_node_feats)

class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()
        self.project_edge = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * node_feat_size, 1), nn.LeakyReLU())
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])
        return self.attentive_gru(g, logits, node_feats)

class AttentiveFPGNN(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2, graph_feat_size=200, dropout=Dropout):
        super(AttentiveFPGNN, self).__init__()
        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats

class DGL_AttentiveFP(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers=2, num_timesteps=2, graph_feat_size=200, predictor_dim=None):
        super(DGL_AttentiveFP, self).__init__()
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size, num_layers=num_layers, graph_feat_size=graph_feat_size)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size, num_timesteps=num_timesteps)
        self.transform = nn.Linear(graph_feat_size, predictor_dim)

    def forward(self, bg):
        node_feats = bg.ndata.pop('h')
        edge_feats = bg.edata.pop('e')
        node_feats = self.gnn(bg, node_feats, edge_feats)
        graph_feats = self.readout(bg, node_feats, False)
        return self.transform(graph_feats)

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.dropout_rate = config['dropout_rate']
        self.embedding = nn.Linear(63, config['embedding_size'])
        self.convs_1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=config['embedding_size'], out_channels=config['feature_size'], kernel_size=h, padding=(h - 1) // 2),
                nn.ReLU()
            ) for h in config['window_sizes']
        ])
        self.convs_2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=config['embedding_size'] * len(config['window_sizes']), out_channels=config['feature_size'], kernel_size=h, padding=(h - 1) // 2),
                nn.ReLU()
            ) for h in config['window_sizes']
        ])
        self.convs_3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=config['embedding_size'] * len(config['window_sizes']), out_channels=config['feature_size'], kernel_size=h, padding=math.floor((h - 1) / 2)),
                nn.ReLU()
            ) for h in config['window_sizes']
        ])
        self.MaxPool = nn.MaxPool1d(kernel_size=config['max_text_len'] - config['window_sizes'][-1] + 1)
        self.fc = nn.Linear(config['feature_size'] * len(config['window_sizes']), config['feature_size'])

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x = embed_x.permute(0, 2, 1)
        out1 = torch.cat([conv(embed_x) for conv in self.convs_1], dim=1)
        out2 = torch.cat([conv(out1) for conv in self.convs_2], dim=1) + out1
        out3 = torch.cat([conv(out2) for conv in self.convs_3], dim=1) + out2
        out3 = self.MaxPool(out3).squeeze(-1)
        return F.dropout(self.fc(out3), p=self.dropout_rate)


class ConvNCF(nn.Module):
    def __init__(self, ad_dim, drug_embed_dim, side_embed_dim, hid_embed_dim, dropout, **base_config_TextCNN):
        super(ConvNCF, self).__init__()
        self.drugs_feature_dim = drug_embed_dim
        self.embed_dim = hid_embed_dim
        self.sides_dim = side_embed_dim
        self.FP_model = FingerprintCNN(fp_dim=2048, hidden_dim=hid_embed_dim)
        self.CNN_model = TextCNN(base_config_TextCNN)
        self.GCN_model = DGL_AttentiveFP(node_feat_size=39, edge_feat_size=11, num_layers=3, num_timesteps=2,
                                         graph_feat_size=64, predictor_dim=hid_embed_dim)
        self.dropout = dropout

        self.alpha = 0.95
        self.temperature = 0.5
        self.consistency_lambda = 0.2
        self.register_buffer('ema_weights', torch.ones(7) / 7 + torch.randn(7) * 0.01)
        self.register_buffer('raw_weights', torch.zeros(7))


        self.drugs_layer = nn.Linear(self.drugs_feature_dim, self.embed_dim)
        self.drugs_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


        self.sides_layer1 = nn.Linear(self.sides_dim[0], self.embed_dim)
        self.sides_layer_1_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.sides_bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


        self.drugs_layer_label = nn.Linear(ad_dim[0], self.embed_dim)
        self.drugs_layer_label_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_label_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.drugs_layer_f = nn.Linear(ad_dim[0], self.embed_dim)
        self.drugs_layer_f_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_f_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.sides_layer_label = nn.Linear(ad_dim[1], self.embed_dim)
        self.sides_layer_label_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.sides_label_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)


        self.classifier1 = nn.Linear(7 * self.embed_dim, self.embed_dim)
        nn.init.xavier_uniform_(self.classifier1.weight)
        self.classifier_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.multi_classifier1 = nn.Linear(7 * self.embed_dim, self.embed_dim)
        self.multi_classifier_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.classifier2 = nn.Linear(self.embed_dim, 2)
        self.multi_classifier2 = nn.Linear(self.embed_dim, 1)


        self.attention = multimodal_Attention(self.embed_dim, 8, dropout, dropout)


        self.current_weights = None

    def cosine_similarity(self, x, y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return (x * y).sum(dim=1).mean()

    def forward(self, drug_smiles_graph, drug_smiles, drug_fp, drug_features, side_feature1, side_feature2,
                batch_drug_label, batch_side_label, device):

        x_graph = self.GCN_model(dgl.batch(list(drug_smiles_graph)).to(device))


        x_simles = self.CNN_model(drug_smiles.to(device))


        x_fp = self.FP_model(drug_fp.to(device))


        x_side1 = F.relu(self.sides_bn1(self.sides_layer1(side_feature1.to(device))), inplace=True)
        x_side1 = F.dropout(x_side1, training=self.training, p=self.dropout)
        x_side1 = self.sides_layer_1_1(x_side1)


        x_label_drug1 = F.relu(self.drugs_label_bn(self.drugs_layer_label(batch_drug_label[0].to(device))),
                               inplace=True)
        x_label_drug1 = F.dropout(x_label_drug1, training=self.training, p=self.dropout)
        x_label_drug1 = self.drugs_layer_label_1(x_label_drug1)


        x_label_side1 = F.relu(self.sides_label_bn(self.sides_layer_label(batch_side_label[0].to(device))),
                               inplace=True)
        x_label_side1 = F.dropout(x_label_side1, training=self.training, p=self.dropout)
        x_label_side1 = self.sides_layer_label_1(x_label_side1)

        x_label_drug2 = F.relu(self.drugs_f_bn(self.drugs_layer_f(batch_drug_label[1].to(device))), inplace=True)
        x_label_drug2 = F.dropout(x_label_drug2, training=self.training, p=self.dropout)
        x_label_drug2 = self.drugs_layer_f_1(x_label_drug2)


        modalities = [x_graph, x_simles, x_fp, x_label_drug1, x_label_drug2, x_label_side1, x_side1]
        modalities = [F.normalize(mod, p=2, dim=-1) for mod in modalities]


        modalities = [x_graph, x_simles, x_fp, x_label_drug1,
                      x_label_drug2, x_label_side1, x_side1]
        modalities = [F.normalize(mod, p=2, dim=-1) for mod in modalities]


        n_modalities = len(modalities)
        similarity_matrix = torch.zeros((n_modalities, n_modalities), device=device)

        for i in range(n_modalities):
            for j in range(n_modalities):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = F.cosine_similarity(modalities[i], modalities[j], dim=1).mean()
                    similarity_matrix[i, j] = torch.clamp(sim, min=0.0, max=1.0)



        importance_scores = similarity_matrix.mean(dim=1)
        weights = F.softmax(importance_scores / self.temperature, dim=0)


        raw_weights = F.softmax(importance_scores / self.temperature, dim=0)
        ema_weights = self.alpha * self.ema_weights + (1 - self.alpha) * raw_weights
        ema_weights = ema_weights / (ema_weights.sum() + 1e-9)


        self.ema_weights.copy_(ema_weights.detach())
        self.current_weights = ema_weights.detach()

        consistency_loss = sum(
            F.mse_loss(modalities[i], modalities[j])
            for i in range(n_modalities)
            for j in range(i + 1, n_modalities)
        )
        self.consistency_loss = self.consistency_lambda * consistency_loss


        weighted_modalities = [mod * weight for mod, weight in zip(modalities, weights)]
        total1 = torch.stack(weighted_modalities, dim=1)
        attention_output1 = self.attention(total1)
        total_1 = attention_output1.view(attention_output1.shape[0], -1)

        classification = F.relu(self.classifier_bn(self.classifier1(total_1)), inplace=True)
        classification = F.dropout(classification, training=self.training, p=self.dropout)
        classification = self.classifier2(classification)

        multi_class = F.relu(self.multi_classifier_bn(self.multi_classifier1(total_1)), inplace=True)
        multi_class = F.dropout(multi_class, training=self.training, p=self.dropout)
        multi_class = self.multi_classifier2(multi_class)

        return classification.squeeze(-1), multi_class.squeeze(-1)
