import math
import torch
import torch.nn as nn
import dgl
from torchlibrosa.augmentation import SpecAugmentation
from framework.gated_gcn_layer import GatedGCNLayer
import torch.nn.functional as F


def move_data_to_gpu(x, cuda, half=False):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")
    if cuda:
        x = x.cuda()
        if half:
            x = x.half()
    return x



def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x




def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class CrossTransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 nhead,
                 drop=0.1):
        super(CrossTransformerEncoder, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(input_dim, d_model, bias=False)
        self.k_proj = nn.Linear(input_dim, d_model, bias=False)
        self.v_proj = nn.Linear(input_dim, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, input_dim, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.Dropout(drop),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, node_num, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        dim2 = int(x.size()[0]/node_num)
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))
        message = message.reshape(-1, dim2, self.dim)
        message = self.norm1(message)
        x = x.reshape(-1, dim2, self.dim)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message



class MLGL(nn.Module):
    def __init__(self, event_num, hidden_dim, out_dim, in_dim,
                 n_layers, emb_dim,
                 ):

        super(MLGL, self).__init__()
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.spec = False

        self.event_num = event_num

        self.conv_block1_ar1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_ar1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_ar1 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e24 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e24 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e24 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e7 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e7 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e7 = ConvBlock(in_channels=128, out_channels=256)
        node_emb_dim = emb_dim

        pann_dim = 256
        self.coarse_num = 7
        self.rate_num = 1
        self.fine_num = 24
        coarse_num = self.coarse_num
        rate_num = self.rate_num
        fine_num = self.fine_num
        self.level1_each_node_emb_node24layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(fine_num)])
        self.level1_node_24_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level2_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level3_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])

        self.level1_each_node_emb_node7layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(coarse_num)])
        self.level1_node_7_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level2_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level3_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])

        self.level1_ar_node_embed = nn.Linear(pann_dim, node_emb_dim, bias=True)
        self.level1_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level2_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level3_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)

        self.emb_dim = emb_dim

        input_dim = 1
        self.att_merge_node_embed_event24 = CrossTransformerEncoder(input_dim, node_emb_dim, nhead=1)
        self.att_merge_node_embed_event7 = CrossTransformerEncoder(input_dim, node_emb_dim, nhead=1)
        self.att_merge_node_embed_ar = CrossTransformerEncoder(input_dim, node_emb_dim, nhead=1)
        ##################################### gnn ####################################################################
        in_dim = in_dim
        in_dim_edge = in_dim

        dropout = 0  # 0.1

        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True

        self.out_dim = out_dim

        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                          self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_24_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                         self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_24_7 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))


    def sub_graph(self, event_embs, max_node_num, input_batch_graph, embedding_h_layer, embedding_e_layer, GCN_layers):
        batched_graph = []
        for each_num in range(event_embs.size()[1]):
            h = event_embs[:, each_num, :]
            g = input_batch_graph[each_num].to('cuda:0')
            g.ndata['feat'] = h
            batched_graph.append(g)

        batched_graph = dgl.batch(batched_graph)
        batch_edges = batched_graph.edata['feat']
        batch_nodes = batched_graph.ndata['feat']

        h = embedding_h_layer(batch_nodes)
        e = embedding_e_layer(batch_edges)

        # convnets
        for conv in GCN_layers:
            h, e, mini_graph = conv(batched_graph, h, e)

        x = h.view(-1, max_node_num, self.out_dim)

        return x.permute(1, 0, 2)

    def mean_max_pooling(self, x_clip_e24):
        x_clip_e24 = torch.mean(x_clip_e24, dim=3)
        (x1_clip_e24, _) = torch.max(x_clip_e24, dim=2)
        x2_clip_e24 = torch.mean(x_clip_e24, dim=2)
        x_clip_e24 = x1_clip_e24 + x2_clip_e24
        return x_clip_e24

    def separately_map_event_emb_to_final_output(self, all_events_num, embeddings_stack, each_classification_layer):
        level1_E24_relu = []
        for event_num in range(all_events_num):
            input_emb = embeddings_stack[event_num]
            level1_E24_relu.append(each_classification_layer[event_num](input_emb))
        level1_E24_relu = torch.cat(level1_E24_relu, dim=-1)
        return level1_E24_relu

    def forward(self, input, batch_graph_24_1,
                                                                      batch_graph_7_1,
                                                                      batch_graph_24_7,
                                                                      batch_graph_24_7_1):

        coarse_num = 7
        rate_num = 1
        fine_num = 24

        (_, seq_len, mel_bins) = input.shape
        x = input[:, None, :, :]

        if self.training and self.spec:
            x = self.spec_augmenter(x)

        x_clip_e24 = self.conv_block1_e24(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block2_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block3_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)

        x_clip_e24 = self.mean_max_pooling(x_clip_e24)

        x_clip_e24 = F.dropout(x_clip_e24, p=0.5, training=self.training)
        x_clip_e24_embeddings = [F.relu_(each_layer(x_clip_e24)) for each_layer in
                                 self.level1_each_node_emb_node24layers]
        x_clip_e24_embeddings_stack = torch.stack(x_clip_e24_embeddings)

        level1_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num, x_clip_e24_embeddings_stack,
                                                                        self.level1_node_24_each_classification_layers)


        # --------------------- level1 event 7 nodes -----------------------------------------------------------------
        x_clip_e7 = self.conv_block1_e7(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block2_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block3_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.mean_max_pooling(x_clip_e7)

        x_clip_e7 = F.dropout(x_clip_e7, p=0.5, training=self.training)
        x_clip_e7_embeddings = [F.relu_(each_layer(x_clip_e7)) for each_layer in self.level1_each_node_emb_node7layers]
        x_clip_e7_embeddings_stack = torch.stack(x_clip_e7_embeddings)

        level1_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num, x_clip_e7_embeddings_stack,
                                                                       self.level1_node_7_each_classification_layers)


        # --------------------- level1 ar node --------------------------------------------------------------------
        x_clip_ar1 = self.conv_block1_ar1(x, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block2_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block3_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.mean_max_pooling(x_clip_ar1)

        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.5, training=self.training)
        x_clip_ar1_embeddings_stack = F.relu_(self.level1_ar_node_embed(x_clip_ar1))
        level1_ar_linear = self.level1_ar_node_each_classification_layers(x_clip_ar1_embeddings_stack)

        # ----------------------------------leve 2 --------------------------------------------------------------------
        event_24_1_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)

        x_24_1_node = self.sub_graph(event_embs=event_24_1_embs, max_node_num=self.fine_num + self.rate_num,
                                     input_batch_graph=batch_graph_24_1,
                                     embedding_h_layer=self.embedding_h_24_1,
                                     embedding_e_layer=self.embedding_e_24_1,
                                     GCN_layers=self.GCN_layers_24_1)
        x_24_node_from_24_1 = x_24_1_node[:fine_num]
        x_1_node_from_24_1 = x_24_1_node[-1]
        level2_e24_node_from_24_1 = x_24_node_from_24_1.permute(1, 0, 2)
        level2_ar1_node_from_24_1 = x_1_node_from_24_1
        event_7_1_embs = torch.cat([x_clip_e7_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)

        x_7_1_node = self.sub_graph(event_embs=event_7_1_embs, max_node_num=coarse_num + rate_num,

                                    input_batch_graph=batch_graph_7_1,
                                    embedding_h_layer=self.embedding_h_7_1,
                                    embedding_e_layer=self.embedding_e_7_1,
                                    GCN_layers=self.GCN_layers_7_1)
        x_7_node_from_7_1 = x_7_1_node[:coarse_num]
        x_1_node_from_7_1 = x_7_1_node[-1]

        level2_e7_node_from_7_1 = x_7_node_from_7_1.permute(1, 0, 2)
        level2_ar1_node_from_7_1 = x_1_node_from_7_1
        # """-------------------------------------------------------------------------------------------------------"""
        event_24_7_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_e7_embeddings_stack], dim=0)
        x_24_7_node = self.sub_graph(event_embs=event_24_7_embs, max_node_num=fine_num + coarse_num,
                                     input_batch_graph=batch_graph_24_7,
                                     embedding_h_layer=self.embedding_h_24_7,
                                     embedding_e_layer=self.embedding_e_24_7,
                                     GCN_layers=self.GCN_layers_24_7)
        x_24_node_from_24_7 = x_24_7_node[:fine_num]
        x_7_node_from_24_7 = x_24_7_node[fine_num:fine_num + coarse_num]

        level2_e24_node_from_24_7 = x_24_node_from_24_7.permute(1, 0, 2)
        level2_e7_node_from_24_7 = x_7_node_from_24_7.permute(1, 0, 2)
        level2_e24_node_embeddings_stack = self.att_merge_node_embed_event24(
            x_24_node_from_24_1.reshape(-1, self.emb_dim, 1),
            x_24_node_from_24_7.reshape(-1, self.emb_dim, 1),
            self.fine_num)
        level2_e7_node_embeddings_stack = self.att_merge_node_embed_event7(
            x_7_node_from_7_1.reshape(-1, self.emb_dim, 1),
            x_7_node_from_24_7.reshape(-1, self.emb_dim, 1),

            self.coarse_num)
        level2_ar_node_embeddings_stack = self.att_merge_node_embed_ar(x_1_node_from_7_1[:, :, None],
                                                                       x_1_node_from_24_1[:, :, None],

                                                                       1)
        level2_ar_node_embeddings_stack = level2_ar_node_embeddings_stack[0]
        # -------------------------------------------------------------------------------------------------------------

        level2_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        level2_e24_node_embeddings_stack,
                                                                        self.level2_node_24_each_classification_layers)

        level2_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       level2_e7_node_embeddings_stack,
                                                                       self.level2_node_7_each_classification_layers)
        level2_ar_linear = self.level2_ar_node_each_classification_layers(level2_ar_node_embeddings_stack)
        level2_E24_relu += level1_E24_relu
        level2_E7_relu += level1_E7_relu
        level2_ar_linear += level1_ar_linear
        event_24_7_1_embs = torch.cat([level2_e24_node_embeddings_stack, level2_e7_node_embeddings_stack,
                                       level2_ar_node_embeddings_stack[None, :, :]], dim=0)
        x_24_7_1_node = self.sub_graph(event_embs=event_24_7_1_embs, max_node_num=fine_num + coarse_num + rate_num,
                                       # input_batch_graph=self.batch_graph_24_7_1,
                                       input_batch_graph=batch_graph_24_7_1,
                                       embedding_h_layer=self.embedding_h_24_7_1,
                                       embedding_e_layer=self.embedding_e_24_7_1,
                                       GCN_layers=self.GCN_layers_24_7_1)
        node_ar_level3 = x_24_7_1_node[-1]
        node_24_level3 = x_24_7_1_node[:self.fine_num]
        node_7_level3 = x_24_7_1_node[self.fine_num:self.fine_num + self.coarse_num]

        level3_e24_node = node_24_level3.permute(1, 0, 2)
        level3_e7_node = node_7_level3.permute(1, 0, 2)
        level3_ar1_node = node_ar_level3

        level3_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        node_24_level3,
                                                                        self.level3_node_24_each_classification_layers)
        level3_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       node_7_level3,
                                                                       self.level3_node_7_each_classification_layers)
        level3_ar_linear = self.level3_ar_node_each_classification_layers(node_ar_level3)
        level3_E24_relu += level2_E24_relu
        level3_E7_relu += level2_E7_relu
        level3_ar_linear += level2_ar_linear
        return level1_E24_relu, level1_E7_relu, level1_ar_linear, level2_E24_relu, level2_E7_relu, level2_ar_linear, \
               level3_E24_relu, level3_E7_relu, level3_ar_linear



class MLGL_addition(nn.Module):
    def __init__(self, event_num, hidden_dim, out_dim, in_dim, n_layers, emb_dim, ):

        super(MLGL_addition, self).__init__()
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.spec = False

        self.event_num = event_num

        self.conv_block1_ar1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_ar1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_ar1 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e24 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e24 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e24 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e7 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e7 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e7 = ConvBlock(in_channels=128, out_channels=256)

        node_emb_dim = emb_dim

        pann_dim = 256
        self.coarse_num = 7
        self.rate_num = 1
        self.fine_num = 24
        coarse_num = self.coarse_num
        rate_num = self.rate_num
        fine_num = self.fine_num
        self.level1_each_node_emb_node24layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(fine_num)])
        self.level1_node_24_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level2_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level3_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])

        self.level1_each_node_emb_node7layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(coarse_num)])
        self.level1_node_7_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level2_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level3_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])

        self.level1_ar_node_embed = nn.Linear(pann_dim, node_emb_dim, bias=True)
        self.level1_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level2_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level3_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)

        self.emb_dim = emb_dim
        ##################################### gnn ####################################################################
        in_dim = in_dim  # 1  # 527
        in_dim_edge = in_dim  # 1  # 527

        dropout = 0  # 0.1

        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True

        self.out_dim = out_dim

        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                          self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_24_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                         self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_24_7 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.init_weight()

    def init_weight(self):

        for i in range(self.event_num):
            init_layer(self.level1_node_24_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node24layers[i])

        for i in range(self.coarse_num):
            init_layer(self.level1_node_7_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node7layers[i])


    def sub_graph(self, event_embs, max_node_num, input_batch_graph, embedding_h_layer, embedding_e_layer, GCN_layers):
        batched_graph = []
        for each_num in range(event_embs.size()[1]):
            h = event_embs[:, each_num, :]
            g = input_batch_graph[each_num].to('cuda:0')
            g.ndata['feat'] = h
            batched_graph.append(g)

        batched_graph = dgl.batch(batched_graph)
        batch_edges = batched_graph.edata['feat']
        batch_nodes = batched_graph.ndata['feat']

        h = embedding_h_layer(batch_nodes)
        e = embedding_e_layer(batch_edges)

        # convnets
        for conv in GCN_layers:
            h, e, mini_graph = conv(batched_graph, h, e)

        x = h.view(-1, max_node_num, self.out_dim)

        return x.permute(1, 0, 2)

    def mean_max_pooling(self, x_clip_e24):
        x_clip_e24 = torch.mean(x_clip_e24, dim=3)
        (x1_clip_e24, _) = torch.max(x_clip_e24, dim=2)
        x2_clip_e24 = torch.mean(x_clip_e24, dim=2)
        x_clip_e24 = x1_clip_e24 + x2_clip_e24
        return x_clip_e24

    def separately_map_event_emb_to_final_output(self, all_events_num, embeddings_stack, each_classification_layer):
        level1_E24_relu = []
        for event_num in range(all_events_num):
            input_emb = embeddings_stack[event_num]
            level1_E24_relu.append(each_classification_layer[event_num](input_emb))
        level1_E24_relu = torch.cat(level1_E24_relu, dim=-1)
        return level1_E24_relu

    def forward(self, input, batch_graph_24_1, batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1):

        coarse_num = 7
        rate_num = 1
        fine_num = 24

        (_, seq_len, mel_bins) = input.shape
        x = input[:, None, :, :]

        if self.training and self.spec:
            x = self.spec_augmenter(x)

        # --------------------- level1 event 24 nodes -----------------------------------------------------------------
        x_clip_e24 = self.conv_block1_e24(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block2_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block3_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)

        x_clip_e24 = self.mean_max_pooling(x_clip_e24)

        x_clip_e24 = F.dropout(x_clip_e24, p=0.5, training=self.training)
        x_clip_e24_embeddings = [F.relu_(each_layer(x_clip_e24)) for each_layer in self.level1_each_node_emb_node24layers]  # embeddings _e24
        x_clip_e24_embeddings_stack = torch.stack(x_clip_e24_embeddings)
        level1_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num, x_clip_e24_embeddings_stack,
                                                                        self.level1_node_24_each_classification_layers)


        # --------------------- level1 event 7 nodes -----------------------------------------------------------------
        x_clip_e7 = self.conv_block1_e7(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block2_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block3_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)

        x_clip_e7 = self.mean_max_pooling(x_clip_e7)

        x_clip_e7 = F.dropout(x_clip_e7, p=0.5, training=self.training)
        x_clip_e7_embeddings = [F.relu_(each_layer(x_clip_e7)) for each_layer in self.level1_each_node_emb_node7layers]
        x_clip_e7_embeddings_stack = torch.stack(x_clip_e7_embeddings)

        level1_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num, x_clip_e7_embeddings_stack,
                                                                        self.level1_node_7_each_classification_layers)


        # --------------------- level1 ar node --------------------------------------------------------------------
        x_clip_ar1 = self.conv_block1_ar1(x, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block2_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block3_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)

        x_clip_ar1 = self.mean_max_pooling(x_clip_ar1)

        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.5, training=self.training)
        x_clip_ar1_embeddings_stack = F.relu_(self.level1_ar_node_embed(x_clip_ar1))
        level1_ar_linear = self.level1_ar_node_each_classification_layers(x_clip_ar1_embeddings_stack)

        # ----------------------------------leve 2 --------------------------------------------------------------------
        event_24_1_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)

        x_24_1_node = self.sub_graph(event_embs=event_24_1_embs, max_node_num=self.fine_num + self.rate_num,
                                                 input_batch_graph=batch_graph_24_1,
                                                 embedding_h_layer=self.embedding_h_24_1,
                                                 embedding_e_layer=self.embedding_e_24_1,
                                                 GCN_layers=self.GCN_layers_24_1)
        x_24_node_from_24_1 = x_24_1_node[:fine_num]
        x_1_node_from_24_1 = x_24_1_node[-1]
        """-------------------------------------------------------------------------------------------------------"""

        event_7_1_embs = torch.cat([x_clip_e7_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)
        x_7_1_node = self.sub_graph(event_embs=event_7_1_embs, max_node_num=coarse_num + rate_num,
                                             input_batch_graph=batch_graph_7_1,
                                             embedding_h_layer=self.embedding_h_7_1,
                                             embedding_e_layer=self.embedding_e_7_1,
                                             GCN_layers=self.GCN_layers_7_1)
        x_7_node_from_7_1 = x_7_1_node[:coarse_num]
        x_1_node_from_7_1 = x_7_1_node[-1]
        # """-------------------------------------------------------------------------------------------------------"""
        event_24_7_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_e7_embeddings_stack], dim=0)
        x_24_7_node = self.sub_graph(event_embs=event_24_7_embs, max_node_num=fine_num + coarse_num,
                                             input_batch_graph=batch_graph_24_7,
                                             embedding_h_layer=self.embedding_h_24_7,
                                             embedding_e_layer=self.embedding_e_24_7,
                                             GCN_layers=self.GCN_layers_24_7)
        x_24_node_from_24_7 = x_24_7_node[:fine_num]
        x_7_node_from_24_7 = x_24_7_node[fine_num:fine_num+coarse_num]
        level2_e24_node_embeddings_stack = x_24_node_from_24_1 + x_24_node_from_24_7
        level2_e7_node_embeddings_stack = x_7_node_from_7_1 + x_7_node_from_24_7
        level2_ar_node_embeddings_stack = x_1_node_from_7_1 + x_1_node_from_24_1
        level2_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        level2_e24_node_embeddings_stack,
                                                                        self.level2_node_24_each_classification_layers)

        level2_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       level2_e7_node_embeddings_stack,
                                                                       self.level2_node_7_each_classification_layers)
        level2_ar_linear = self.level2_ar_node_each_classification_layers(level2_ar_node_embeddings_stack)  # 回归不要用relu

        event_24_7_1_embs = torch.cat([level2_e24_node_embeddings_stack, level2_e7_node_embeddings_stack,
                                       level2_ar_node_embeddings_stack[None, :, :]], dim=0)
        x_24_7_1_node = self.sub_graph(event_embs=event_24_7_1_embs, max_node_num=fine_num+coarse_num+rate_num,
                                                 input_batch_graph=batch_graph_24_7_1,
                              embedding_h_layer=self.embedding_h_24_7_1, embedding_e_layer=self.embedding_e_24_7_1,
                              GCN_layers=self.GCN_layers_24_7_1)
        node_ar_level3 = x_24_7_1_node[-1]
        node_24_level3 = x_24_7_1_node[:self.fine_num]
        node_7_level3 = x_24_7_1_node[self.fine_num:self.fine_num+self.coarse_num]

        level3_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        node_24_level3,
                                                                        self.level3_node_24_each_classification_layers)

        level3_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       node_7_level3,
                                                                       self.level3_node_7_each_classification_layers)

        level3_ar_linear = self.level3_ar_node_each_classification_layers(node_ar_level3)
        level3_E24_relu += level1_E24_relu
        level3_E7_relu += level1_E7_relu
        level3_ar_linear += level1_ar_linear
        return level1_E24_relu, level1_E7_relu, level1_ar_linear, level2_E24_relu, level2_E7_relu, level2_ar_linear, \
               level3_E24_relu, level3_E7_relu, level3_ar_linear


class MLGL_concate(nn.Module):
    def __init__(self, event_num, hidden_dim, out_dim, in_dim,
                 n_layers, emb_dim,
                 ):

        super(MLGL_concate, self).__init__()
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.spec = False

        self.event_num = event_num

        self.conv_block1_ar1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_ar1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_ar1 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e24 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e24 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e24 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e7 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e7 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e7 = ConvBlock(in_channels=128, out_channels=256)

        # 由于后面sub_graph 里面 的维度是 64，所以这里还是默认64 吧
        node_emb_dim = emb_dim

        pann_dim = 256
        self.coarse_num = 7
        self.rate_num = 1
        self.fine_num = 24
        coarse_num = self.coarse_num
        rate_num = self.rate_num
        fine_num = self.fine_num
        self.level1_each_node_emb_node24layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(fine_num)])
        self.level1_node_24_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level2_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level3_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])

        self.level1_each_node_emb_node7layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(coarse_num)])
        self.level1_node_7_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level2_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level3_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])

        self.level1_ar_node_embed = nn.Linear(pann_dim, node_emb_dim, bias=True)
        self.level1_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level2_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level3_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)

        self.emb_dim = emb_dim

        self.merge_node_embed_event24 = nn.Linear(node_emb_dim * 2, node_emb_dim)
        self.merge_node_embed_event7 = nn.Linear(node_emb_dim * 2, node_emb_dim)
        self.merge_node_embed_ar = nn.Linear(node_emb_dim * 2, node_emb_dim)

        ##################################### gnn ####################################################################
        in_dim = in_dim  # 1  # 527
        in_dim_edge = in_dim  # 1  # 527

        dropout = 0  # 0.1

        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True

        self.out_dim = out_dim

        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                          self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_24_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                         self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_24_7 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.init_weight()

    def init_weight(self):

        for i in range(self.event_num):
            init_layer(self.level1_node_24_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node24layers[i])

        for i in range(self.coarse_num):
            init_layer(self.level1_node_7_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node7layers[i])


    def sub_graph(self, event_embs, max_node_num, input_batch_graph, embedding_h_layer, embedding_e_layer, GCN_layers):
        batched_graph = []  # dgl.batch(batch_x)
        # print('event_embs: ', event_embs.size())  # event_embs:  torch.Size([24, 64, 64])
        for each_num in range(event_embs.size()[1]):
            h = event_embs[:, each_num, :]
            # print(h.shape)  # torch.Size([24, 64]) (top 24 events, ，每个事件的维度是 64)
            g = input_batch_graph[each_num].to('cuda:0')
            g.ndata['feat'] = h  # 527*1---graph
            batched_graph.append(g)

        batched_graph = dgl.batch(batched_graph)
        batch_edges = batched_graph.edata['feat']
        batch_nodes = batched_graph.ndata['feat']

        h = embedding_h_layer(batch_nodes)  # 点特征
        e = embedding_e_layer(batch_edges)  # 边特征

        # convnets
        for conv in GCN_layers:
            h, e, mini_graph = conv(batched_graph, h, e)

        x = h.view(-1, max_node_num, self.out_dim)

        return x.permute(1, 0, 2)

    def mean_max_pooling(self, x_clip_e24):
        x_clip_e24 = torch.mean(x_clip_e24, dim=3)
        (x1_clip_e24, _) = torch.max(x_clip_e24, dim=2)
        x2_clip_e24 = torch.mean(x_clip_e24, dim=2)
        x_clip_e24 = x1_clip_e24 + x2_clip_e24
        return x_clip_e24

    def separately_map_event_emb_to_final_output(self, all_events_num, embeddings_stack, each_classification_layer):
        level1_E24_relu = []
        for event_num in range(all_events_num):
            input_emb = embeddings_stack[event_num]
            level1_E24_relu.append(each_classification_layer[event_num](input_emb))
        level1_E24_relu = torch.cat(level1_E24_relu, dim=-1)
        return level1_E24_relu

    def forward(self, input, batch_graph_24_1, batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1):

        coarse_num = 7
        rate_num = 1
        fine_num = 24

        (_, seq_len, mel_bins) = input.shape
        x = input[:, None, :, :]

        if self.training and self.spec:
            x = self.spec_augmenter(x)

        # --------------------- level1 event 24 nodes -----------------------------------------------------------------
        x_clip_e24 = self.conv_block1_e24(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block2_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block3_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        # x_clip_e24 = self.conv_block4_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        # x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)

        x_clip_e24 = self.mean_max_pooling(x_clip_e24)
        # print('x_clip: ', x_clip.size())  # 10s clip: torch.Size([128, 2048])

        x_clip_e24 = F.dropout(x_clip_e24, p=0.5, training=self.training)
        x_clip_e24_embeddings = [F.relu_(each_layer(x_clip_e24)) for each_layer in self.level1_each_node_emb_node24layers]  # embeddings _e24
        x_clip_e24_embeddings_stack = torch.stack(x_clip_e24_embeddings)

        level1_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num, x_clip_e24_embeddings_stack,
                                                                        self.level1_node_24_each_classification_layers)
        # print('level1_E24_relu: ', level1_E24_relu.size())   # torch.Size([64, 24])

        # --------------------- level1 event 7 nodes -----------------------------------------------------------------
        x_clip_e7 = self.conv_block1_e7(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block2_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block3_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        # x_clip_e7 = self.conv_block4_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        # x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)

        x_clip_e7 = self.mean_max_pooling(x_clip_e7)

        x_clip_e7 = F.dropout(x_clip_e7, p=0.5, training=self.training)
        x_clip_e7_embeddings = [F.relu_(each_layer(x_clip_e7)) for each_layer in self.level1_each_node_emb_node7layers]
        x_clip_e7_embeddings_stack = torch.stack(x_clip_e7_embeddings)

        level1_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num, x_clip_e7_embeddings_stack,
                                                                        self.level1_node_7_each_classification_layers)
        # print('level1_E7_relu: ', level1_E7_relu.size())

        # --------------------- level1 ar node --------------------------------------------------------------------
        x_clip_ar1 = self.conv_block1_ar1(x, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block2_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block3_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        # x_clip_ar1 = self.conv_block4_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        # x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)

        x_clip_ar1 = self.mean_max_pooling(x_clip_ar1)

        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.5, training=self.training)
        x_clip_ar1_embeddings_stack = F.relu_(self.level1_ar_node_embed(x_clip_ar1))
        level1_ar_linear = self.level1_ar_node_each_classification_layers(x_clip_ar1_embeddings_stack)

        # ----------------------------------leve 2 --------------------------------------------------------------------
        event_24_1_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)
        # print(event_24_1_embs.size())
        x_24_1_node = self.sub_graph(event_embs=event_24_1_embs, max_node_num=self.fine_num + self.rate_num,
                                                 input_batch_graph=batch_graph_24_1,
                                                 embedding_h_layer=self.embedding_h_24_1,
                                                 embedding_e_layer=self.embedding_e_24_1,
                                                 GCN_layers=self.GCN_layers_24_1)
        x_24_node_from_24_1 = x_24_1_node[:fine_num]
        x_1_node_from_24_1 = x_24_1_node[-1]
        """-------------------------------------------------------------------------------------------------------"""

        event_7_1_embs = torch.cat([x_clip_e7_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)
        x_7_1_node = self.sub_graph(event_embs=event_7_1_embs, max_node_num=coarse_num + rate_num,
                                             input_batch_graph=batch_graph_7_1,
                                             embedding_h_layer=self.embedding_h_7_1,
                                             embedding_e_layer=self.embedding_e_7_1,
                                             GCN_layers=self.GCN_layers_7_1)
        x_7_node_from_7_1 = x_7_1_node[:coarse_num]
        x_1_node_from_7_1 = x_7_1_node[-1]
        # """-------------------------------------------------------------------------------------------------------"""
        event_24_7_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_e7_embeddings_stack], dim=0)
        x_24_7_node = self.sub_graph(event_embs=event_24_7_embs, max_node_num=fine_num + coarse_num,
                                             input_batch_graph=batch_graph_24_7,
                                             embedding_h_layer=self.embedding_h_24_7,
                                             embedding_e_layer=self.embedding_e_24_7,
                                             GCN_layers=self.GCN_layers_24_7)
        x_24_node_from_24_7 = x_24_7_node[:fine_num]
        x_7_node_from_24_7 = x_24_7_node[fine_num:fine_num+coarse_num]

        level2_e24_node_embeddings_stack = self.merge_node_embed_event24(torch.cat([x_24_node_from_24_1, x_24_node_from_24_7], dim=-1))
        level2_e7_node_embeddings_stack = self.merge_node_embed_event7(torch.cat([x_7_node_from_24_7, x_7_node_from_7_1], dim=-1))
        level2_ar_node_embeddings_stack = self.merge_node_embed_ar(torch.cat([x_1_node_from_24_1, x_1_node_from_7_1], dim=-1))


        level2_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        level2_e24_node_embeddings_stack,
                                                                        self.level2_node_24_each_classification_layers)

        level2_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       level2_e7_node_embeddings_stack,
                                                                       self.level2_node_7_each_classification_layers)
        level2_ar_linear = self.level2_ar_node_each_classification_layers(level2_ar_node_embeddings_stack)  # 回归不要用relu


        event_24_7_1_embs = torch.cat([level2_e24_node_embeddings_stack, level2_e7_node_embeddings_stack,
                                       level2_ar_node_embeddings_stack[None, :, :]], dim=0)
        x_24_7_1_node = self.sub_graph(event_embs=event_24_7_1_embs, max_node_num=fine_num+coarse_num+rate_num,
                                                 input_batch_graph=batch_graph_24_7_1,
                              embedding_h_layer=self.embedding_h_24_7_1, embedding_e_layer=self.embedding_e_24_7_1,
                              GCN_layers=self.GCN_layers_24_7_1)

        node_ar_level3 = x_24_7_1_node[-1]
        node_24_level3 = x_24_7_1_node[:self.fine_num]
        node_7_level3 = x_24_7_1_node[self.fine_num:self.fine_num+self.coarse_num]

        level3_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        node_24_level3,
                                                                        self.level3_node_24_each_classification_layers)

        level3_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       node_7_level3,
                                                                       self.level3_node_7_each_classification_layers)

        level3_ar_linear = self.level3_ar_node_each_classification_layers(node_ar_level3)

        level3_E24_relu += level1_E24_relu
        level3_E7_relu += level1_E7_relu
        level3_ar_linear += level1_ar_linear

        return level1_E24_relu, level1_E7_relu, level1_ar_linear, level2_E24_relu, level2_E7_relu, level2_ar_linear, \
               level3_E24_relu, level3_E7_relu, level3_ar_linear



class MLGL_Hadamard(nn.Module):
    def __init__(self, event_num, hidden_dim, out_dim, in_dim, n_layers, emb_dim,):

        super(MLGL_Hadamard, self).__init__()
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.spec = False

        self.event_num = event_num

        self.conv_block1_ar1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_ar1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_ar1 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e24 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e24 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e24 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e7 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e7 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e7 = ConvBlock(in_channels=128, out_channels=256)

        node_emb_dim = emb_dim

        pann_dim = 256
        self.coarse_num = 7
        self.rate_num = 1
        self.fine_num = 24
        coarse_num = self.coarse_num
        rate_num = self.rate_num
        fine_num = self.fine_num
        self.level1_each_node_emb_node24layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(fine_num)])
        self.level1_node_24_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level2_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level3_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])

        self.level1_each_node_emb_node7layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(coarse_num)])
        self.level1_node_7_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level2_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level3_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])

        self.level1_ar_node_embed = nn.Linear(pann_dim, node_emb_dim, bias=True)
        self.level1_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level2_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level3_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)

        self.emb_dim = emb_dim

        in_dim = in_dim
        in_dim_edge = in_dim

        dropout = 0  # 0.1

        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True

        self.out_dim = out_dim

        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                          self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_24_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                         self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_24_7 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        ##############################################################################################################


        self.init_weight()

    def init_weight(self):

        for i in range(self.event_num):
            init_layer(self.level1_node_24_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node24layers[i])

        for i in range(self.coarse_num):
            init_layer(self.level1_node_7_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node7layers[i])


    def sub_graph(self, event_embs, max_node_num, input_batch_graph, embedding_h_layer, embedding_e_layer, GCN_layers):
        batched_graph = []
        for each_num in range(event_embs.size()[1]):
            h = event_embs[:, each_num, :]
            g = input_batch_graph[each_num].to('cuda:0')
            g.ndata['feat'] = h
            batched_graph.append(g)

        batched_graph = dgl.batch(batched_graph)
        batch_edges = batched_graph.edata['feat']
        batch_nodes = batched_graph.ndata['feat']

        h = embedding_h_layer(batch_nodes)
        e = embedding_e_layer(batch_edges)

        # convnets
        for conv in GCN_layers:
            h, e, mini_graph = conv(batched_graph, h, e)

        x = h.view(-1, max_node_num, self.out_dim)

        return x.permute(1, 0, 2)

    def mean_max_pooling(self, x_clip_e24):
        x_clip_e24 = torch.mean(x_clip_e24, dim=3)
        (x1_clip_e24, _) = torch.max(x_clip_e24, dim=2)
        x2_clip_e24 = torch.mean(x_clip_e24, dim=2)
        x_clip_e24 = x1_clip_e24 + x2_clip_e24
        return x_clip_e24

    def separately_map_event_emb_to_final_output(self, all_events_num, embeddings_stack, each_classification_layer):
        level1_E24_relu = []
        for event_num in range(all_events_num):
            input_emb = embeddings_stack[event_num]
            level1_E24_relu.append(each_classification_layer[event_num](input_emb))
        level1_E24_relu = torch.cat(level1_E24_relu, dim=-1)
        return level1_E24_relu

    def forward(self, input, batch_graph_24_1, batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1):

        coarse_num = 7
        rate_num = 1
        fine_num = 24

        (_, seq_len, mel_bins) = input.shape
        x = input[:, None, :, :]

        if self.training and self.spec:
            x = self.spec_augmenter(x)

        # --------------------- level1 event 24 nodes -----------------------------------------------------------------
        x_clip_e24 = self.conv_block1_e24(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block2_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block3_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)

        x_clip_e24 = self.mean_max_pooling(x_clip_e24)

        x_clip_e24 = F.dropout(x_clip_e24, p=0.5, training=self.training)
        x_clip_e24_embeddings = [F.relu_(each_layer(x_clip_e24)) for each_layer in self.level1_each_node_emb_node24layers]  # embeddings _e24
        x_clip_e24_embeddings_stack = torch.stack(x_clip_e24_embeddings)

        level1_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num, x_clip_e24_embeddings_stack,
                                                                        self.level1_node_24_each_classification_layers)

        # --------------------- level1 event 7 nodes -----------------------------------------------------------------
        x_clip_e7 = self.conv_block1_e7(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block2_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block3_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)

        x_clip_e7 = self.mean_max_pooling(x_clip_e7)

        x_clip_e7 = F.dropout(x_clip_e7, p=0.5, training=self.training)
        x_clip_e7_embeddings = [F.relu_(each_layer(x_clip_e7)) for each_layer in self.level1_each_node_emb_node7layers]
        x_clip_e7_embeddings_stack = torch.stack(x_clip_e7_embeddings)

        level1_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num, x_clip_e7_embeddings_stack,
                                                                        self.level1_node_7_each_classification_layers)


        # --------------------- level1 ar node --------------------------------------------------------------------
        x_clip_ar1 = self.conv_block1_ar1(x, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block2_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block3_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.mean_max_pooling(x_clip_ar1)

        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.5, training=self.training)
        x_clip_ar1_embeddings_stack = F.relu_(self.level1_ar_node_embed(x_clip_ar1))

        level1_ar_linear = self.level1_ar_node_each_classification_layers(x_clip_ar1_embeddings_stack)

        # ----------------------------------leve 2 --------------------------------------------------------------------
        event_24_1_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)

        x_24_1_node = self.sub_graph(event_embs=event_24_1_embs, max_node_num=self.fine_num + self.rate_num,
                                                 input_batch_graph=batch_graph_24_1,
                                                 embedding_h_layer=self.embedding_h_24_1,
                                                 embedding_e_layer=self.embedding_e_24_1,
                                                 GCN_layers=self.GCN_layers_24_1)
        x_24_node_from_24_1 = x_24_1_node[:fine_num]
        x_1_node_from_24_1 = x_24_1_node[-1]
        """-------------------------------------------------------------------------------------------------------"""

        event_7_1_embs = torch.cat([x_clip_e7_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)

        x_7_1_node = self.sub_graph(event_embs=event_7_1_embs, max_node_num=coarse_num + rate_num,
                                             input_batch_graph=batch_graph_7_1,
                                             embedding_h_layer=self.embedding_h_7_1,
                                             embedding_e_layer=self.embedding_e_7_1,
                                             GCN_layers=self.GCN_layers_7_1)
        x_7_node_from_7_1 = x_7_1_node[:coarse_num]
        x_1_node_from_7_1 = x_7_1_node[-1]
        # """-------------------------------------------------------------------------------------------------------"""
        event_24_7_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_e7_embeddings_stack], dim=0)
        x_24_7_node = self.sub_graph(event_embs=event_24_7_embs, max_node_num=fine_num + coarse_num,
                                             input_batch_graph=batch_graph_24_7,
                                             embedding_h_layer=self.embedding_h_24_7,
                                             embedding_e_layer=self.embedding_e_24_7,
                                             GCN_layers=self.GCN_layers_24_7)
        x_24_node_from_24_7 = x_24_7_node[:fine_num]
        x_7_node_from_24_7 = x_24_7_node[fine_num:fine_num+coarse_num]
        level2_e24_node_embeddings_stack = x_24_node_from_24_1 * x_24_node_from_24_7
        level2_e7_node_embeddings_stack = x_7_node_from_7_1 * x_7_node_from_24_7
        level2_ar_node_embeddings_stack = x_1_node_from_7_1 * x_1_node_from_24_1

        level2_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        level2_e24_node_embeddings_stack,
                                                                        self.level2_node_24_each_classification_layers)

        level2_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       level2_e7_node_embeddings_stack,
                                                                       self.level2_node_7_each_classification_layers)
        level2_ar_linear = self.level2_ar_node_each_classification_layers(level2_ar_node_embeddings_stack)  # 回归不要用relu

        event_24_7_1_embs = torch.cat([level2_e24_node_embeddings_stack, level2_e7_node_embeddings_stack,
                                       level2_ar_node_embeddings_stack[None, :, :]], dim=0)
        x_24_7_1_node = self.sub_graph(event_embs=event_24_7_1_embs, max_node_num=fine_num+coarse_num+rate_num,
                                                 input_batch_graph=batch_graph_24_7_1,
                              embedding_h_layer=self.embedding_h_24_7_1, embedding_e_layer=self.embedding_e_24_7_1,
                              GCN_layers=self.GCN_layers_24_7_1)

        node_ar_level3 = x_24_7_1_node[-1]
        node_24_level3 = x_24_7_1_node[:self.fine_num]
        node_7_level3 = x_24_7_1_node[self.fine_num:self.fine_num+self.coarse_num]

        level3_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        node_24_level3,
                                                                        self.level3_node_24_each_classification_layers)

        level3_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       node_7_level3,
                                                                       self.level3_node_7_each_classification_layers)

        level3_ar_linear = self.level3_ar_node_each_classification_layers(node_ar_level3)
        level3_E24_relu += level1_E24_relu
        level3_E7_relu += level1_E7_relu
        level3_ar_linear += level1_ar_linear

        return level1_E24_relu, level1_E7_relu, level1_ar_linear, level2_E24_relu, level2_E7_relu, level2_ar_linear, \
               level3_E24_relu, level3_E7_relu, level3_ar_linear



class MLGL_Gating(nn.Module):
    def __init__(self, event_num, hidden_dim, out_dim, in_dim, n_layers, emb_dim, ):

        super(MLGL_Gating, self).__init__()
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.spec = False

        self.event_num = event_num

        self.conv_block1_ar1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_ar1 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_ar1 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e24 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e24 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e24 = ConvBlock(in_channels=128, out_channels=256)

        self.conv_block1_e7 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_e7 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_e7 = ConvBlock(in_channels=128, out_channels=256)

        node_emb_dim = emb_dim

        pann_dim = 256
        self.coarse_num = 7
        self.rate_num = 1
        self.fine_num = 24
        coarse_num = self.coarse_num
        rate_num = self.rate_num
        fine_num = self.fine_num
        self.level1_each_node_emb_node24layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(fine_num)])
        self.level1_node_24_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level2_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])
        self.level3_node_24_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(fine_num)])

        self.level1_each_node_emb_node7layers = nn.ModuleList([nn.Linear(pann_dim, node_emb_dim, bias=True) for _ in range(coarse_num)])
        self.level1_node_7_each_classification_layers = nn.ModuleList([nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level2_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])
        self.level3_node_7_each_classification_layers = nn.ModuleList(
            [nn.Linear(node_emb_dim, 1, bias=True) for _ in range(coarse_num)])

        self.level1_ar_node_embed = nn.Linear(pann_dim, node_emb_dim, bias=True)
        self.level1_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level2_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)
        self.level3_ar_node_each_classification_layers = nn.Linear(node_emb_dim, 1, bias=True)

        self.emb_dim = emb_dim

        ###################################### semantic class #######################################################

        self.l3_E24_Glu_linear = nn.Linear(fine_num, fine_num)
        self.l3_E24_Glu_sigmoid = nn.Linear(fine_num, fine_num)

        self.l3_E7_Glu_linear = nn.Linear(coarse_num, coarse_num)
        self.l3_E7_Glu_sigmoid = nn.Linear(coarse_num, coarse_num)

        self.l3_ar_Glu_linear = nn.Linear(rate_num, rate_num)
        self.l3_ar_Glu_sigmoid = nn.Linear(rate_num, rate_num)

        self.l2_E24_Glu_linear = nn.Linear(emb_dim, emb_dim)
        self.l2_E24_Glu_sigmoid = nn.Linear(emb_dim, emb_dim)

        self.l2_E7_Glu_linear = nn.Linear(emb_dim, emb_dim)
        self.l2_E7_Glu_sigmoid = nn.Linear(emb_dim, emb_dim)

        self.l2_ar_Glu_linear = nn.Linear(emb_dim, emb_dim)
        self.l2_ar_Glu_sigmoid = nn.Linear(emb_dim, emb_dim)

        in_dim = in_dim
        in_dim_edge = in_dim

        dropout = 0  # 0.1

        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True

        self.out_dim = out_dim

        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                          self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_24_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                         self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.GCN_layers_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.embedding_h_24_7 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        # -------------------------------------------------------------------------------------------------------------
        self.embedding_h_24_7_1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_e_24_7_1 = nn.Linear(in_dim_edge, hidden_dim)
        self.GCN_layers_24_7_1 = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                            self.batch_norm, self.residual) for _ in
                                              range(n_layers - 1)])
        self.GCN_layers_24_7_1.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))

        self.init_weight()

    def init_weight(self):

        for i in range(self.event_num):
            init_layer(self.level1_node_24_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node24layers[i])

        for i in range(self.coarse_num):
            init_layer(self.level1_node_7_each_classification_layers[i])
            init_layer(self.level1_each_node_emb_node7layers[i])


    def sub_graph(self, event_embs, max_node_num, input_batch_graph, embedding_h_layer, embedding_e_layer, GCN_layers):
        batched_graph = []
        for each_num in range(event_embs.size()[1]):
            h = event_embs[:, each_num, :]
            g = input_batch_graph[each_num].to('cuda:0')
            g.ndata['feat'] = h
            batched_graph.append(g)

        batched_graph = dgl.batch(batched_graph)
        batch_edges = batched_graph.edata['feat']
        batch_nodes = batched_graph.ndata['feat']

        h = embedding_h_layer(batch_nodes)
        e = embedding_e_layer(batch_edges)

        # convnets
        for conv in GCN_layers:
            h, e, mini_graph = conv(batched_graph, h, e)
        x = h.view(-1, max_node_num, self.out_dim)

        return x.permute(1, 0, 2)

    def mean_max_pooling(self, x_clip_e24):
        x_clip_e24 = torch.mean(x_clip_e24, dim=3)
        (x1_clip_e24, _) = torch.max(x_clip_e24, dim=2)
        x2_clip_e24 = torch.mean(x_clip_e24, dim=2)
        x_clip_e24 = x1_clip_e24 + x2_clip_e24
        return x_clip_e24

    def separately_map_event_emb_to_final_output(self, all_events_num, embeddings_stack, each_classification_layer):
        level1_E24_relu = []
        for event_num in range(all_events_num):
            input_emb = embeddings_stack[event_num]
            level1_E24_relu.append(each_classification_layer[event_num](input_emb))
        level1_E24_relu = torch.cat(level1_E24_relu, dim=-1)
        return level1_E24_relu



    def forward(self, input, batch_graph_24_1, batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1):

        coarse_num = 7
        rate_num = 1
        fine_num = 24

        (_, seq_len, mel_bins) = input.shape
        x = input[:, None, :, :]

        if self.training and self.spec:
            x = self.spec_augmenter(x)

        # --------------------- level1 event 24 nodes -----------------------------------------------------------------
        x_clip_e24 = self.conv_block1_e24(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block2_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)
        x_clip_e24 = self.conv_block3_e24(x_clip_e24, pool_size=(2, 2), pool_type='avg')
        x_clip_e24 = F.dropout(x_clip_e24, p=0.2, training=self.training)

        x_clip_e24 = self.mean_max_pooling(x_clip_e24)

        x_clip_e24 = F.dropout(x_clip_e24, p=0.5, training=self.training)
        x_clip_e24_embeddings = [F.relu_(each_layer(x_clip_e24)) for each_layer in self.level1_each_node_emb_node24layers]  # embeddings _e24
        x_clip_e24_embeddings_stack = torch.stack(x_clip_e24_embeddings)


        level1_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num, x_clip_e24_embeddings_stack,
                                                                        self.level1_node_24_each_classification_layers)

        x_clip_e7 = self.conv_block1_e7(x, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block2_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.conv_block3_e7(x_clip_e7, pool_size=(2, 2), pool_type='avg')
        x_clip_e7 = F.dropout(x_clip_e7, p=0.2, training=self.training)
        x_clip_e7 = self.mean_max_pooling(x_clip_e7)

        x_clip_e7 = F.dropout(x_clip_e7, p=0.5, training=self.training)
        x_clip_e7_embeddings = [F.relu_(each_layer(x_clip_e7)) for each_layer in self.level1_each_node_emb_node7layers]
        x_clip_e7_embeddings_stack = torch.stack(x_clip_e7_embeddings)

        level1_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num, x_clip_e7_embeddings_stack,
                                                                        self.level1_node_7_each_classification_layers)


        # --------------------- level1 ar node --------------------------------------------------------------------
        x_clip_ar1 = self.conv_block1_ar1(x, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block2_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)
        x_clip_ar1 = self.conv_block3_ar1(x_clip_ar1, pool_size=(2, 2), pool_type='avg')
        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.2, training=self.training)

        x_clip_ar1 = self.mean_max_pooling(x_clip_ar1)

        x_clip_ar1 = F.dropout(x_clip_ar1, p=0.5, training=self.training)
        x_clip_ar1_embeddings_stack = F.relu_(self.level1_ar_node_embed(x_clip_ar1))

        level1_ar_linear = self.level1_ar_node_each_classification_layers(x_clip_ar1_embeddings_stack)


        # ----------------------------------leve 2 --------------------------------------------------------------------
        event_24_1_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)
        # print(event_24_1_embs.size())
        x_24_1_node = self.sub_graph(event_embs=event_24_1_embs, max_node_num=self.fine_num + self.rate_num,
                                                 input_batch_graph=batch_graph_24_1,
                                                 embedding_h_layer=self.embedding_h_24_1,
                                                 embedding_e_layer=self.embedding_e_24_1,
                                                 GCN_layers=self.GCN_layers_24_1)
        x_24_node_from_24_1 = x_24_1_node[:fine_num]
        x_1_node_from_24_1 = x_24_1_node[-1]
        """-------------------------------------------------------------------------------------------------------"""

        event_7_1_embs = torch.cat([x_clip_e7_embeddings_stack, x_clip_ar1_embeddings_stack[None, :, :]], dim=0)

        x_7_1_node = self.sub_graph(event_embs=event_7_1_embs, max_node_num=coarse_num + rate_num,
                                             input_batch_graph=batch_graph_7_1,
                                             embedding_h_layer=self.embedding_h_7_1,
                                             embedding_e_layer=self.embedding_e_7_1,
                                             GCN_layers=self.GCN_layers_7_1)
        x_7_node_from_7_1 = x_7_1_node[:coarse_num]
        x_1_node_from_7_1 = x_7_1_node[-1]
        # """-------------------------------------------------------------------------------------------------------"""
        event_24_7_embs = torch.cat([x_clip_e24_embeddings_stack, x_clip_e7_embeddings_stack], dim=0)
        x_24_7_node = self.sub_graph(event_embs=event_24_7_embs, max_node_num=fine_num + coarse_num,
                                             input_batch_graph=batch_graph_24_7,
                                             embedding_h_layer=self.embedding_h_24_7,
                                             embedding_e_layer=self.embedding_e_24_7,
                                             GCN_layers=self.GCN_layers_24_7)
        x_24_node_from_24_7 = x_24_7_node[:fine_num]
        x_7_node_from_24_7 = x_24_7_node[fine_num:fine_num+coarse_num]

        level2_e24_node_embeddings_stack = self.l2_E24_Glu_linear(x_24_node_from_24_1) * F.sigmoid(
            self.l2_E24_Glu_sigmoid(x_24_node_from_24_7))
        level2_e7_node_embeddings_stack = self.l2_E24_Glu_linear(x_7_node_from_24_7) * F.sigmoid(
            self.l2_E24_Glu_sigmoid(x_7_node_from_7_1))
        level2_ar_node_embeddings_stack = self.l2_ar_Glu_linear(x_1_node_from_24_1) * F.sigmoid(
            self.l2_ar_Glu_sigmoid(x_1_node_from_7_1))

        level2_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        level2_e24_node_embeddings_stack,
                                                                        self.level2_node_24_each_classification_layers)

        level2_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       level2_e7_node_embeddings_stack,
                                                                       self.level2_node_7_each_classification_layers)
        level2_ar_linear = self.level2_ar_node_each_classification_layers(level2_ar_node_embeddings_stack)
        # ----------------------------------leve 3 --------------------------------------------------------------------
        event_24_7_1_embs = torch.cat([level2_e24_node_embeddings_stack, level2_e7_node_embeddings_stack,
                                       level2_ar_node_embeddings_stack[None, :, :]], dim=0)
        x_24_7_1_node = self.sub_graph(event_embs=event_24_7_1_embs, max_node_num=fine_num+coarse_num+rate_num,
                                                 input_batch_graph=batch_graph_24_7_1,
                              embedding_h_layer=self.embedding_h_24_7_1, embedding_e_layer=self.embedding_e_24_7_1,
                              GCN_layers=self.GCN_layers_24_7_1)

        node_ar_level3 = x_24_7_1_node[-1]
        node_24_level3 = x_24_7_1_node[:self.fine_num]
        node_7_level3 = x_24_7_1_node[self.fine_num:self.fine_num+self.coarse_num]

        level3_E24_relu = self.separately_map_event_emb_to_final_output(self.fine_num,
                                                                        node_24_level3,
                                                                        self.level3_node_24_each_classification_layers)

        level3_E7_relu = self.separately_map_event_emb_to_final_output(self.coarse_num,
                                                                       node_7_level3,
                                                                       self.level3_node_7_each_classification_layers)

        level3_ar_linear = self.level3_ar_node_each_classification_layers(node_ar_level3)
        level3_E24_relu += level1_E24_relu
        level3_E7_relu += level1_E7_relu
        level3_ar_linear += level1_ar_linear
        return level1_E24_relu, level1_E7_relu, level1_ar_linear, level2_E24_relu, level2_E7_relu, level2_ar_linear, \
               level3_E24_relu, level3_E7_relu, level3_ar_linear



