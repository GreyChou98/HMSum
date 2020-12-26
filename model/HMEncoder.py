from model.transformer_encoder import NewTransformerEncoder
from model.GAT import ESEGAT
from model.neural import PositionwiseFeedForward, sequence_mask

import torch.nn as nn
import torch

class HMEncoder(nn.Module):
    def __init__(self, args, device, src_embeddings, padding_idx):
        super(HMEncoder, self).__init__()
        self.args = args
        self.device = device
        self.padding_idx = padding_idx
        self._TFembed = nn.Embedding(50, self.args.emb_size) # box=10 , embed_size = 256

        self.para_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                               self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)
        self.cluster_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                               self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)

        self.cluster2para = ESEGAT(in_dim=self.args.emb_size,
                                   out_dim=self.args.emb_size,
                                   num_heads=self.args.heads,
                                   attn_drop_out=self.args.enc_dropout,
                                   ffn_inner_hidden_size=self.args.ff_size,
                                   ffn_drop_out=self.args.enc_dropout,
                                   feat_embed_size=self.args.emb_size,
                                   layerType="E2S")

        self.para2cluster = ESEGAT(in_dim=self.args.emb_size,
                                   out_dim=self.args.emb_size,
                                   num_heads=self.args.heads,
                                   attn_drop_out=self.args.enc_dropout,
                                   ffn_inner_hidden_size=self.args.ff_size,
                                   ffn_drop_out=self.args.enc_dropout,
                                   feat_embed_size=self.args.emb_size,
                                   layerType="S2E")

        self.layer_norm = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(self.args.enc_hidden_size, self.args.ff_size, self.args.enc_dropout)


    def forward(self, graph):
        self.cnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        self.pnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        src = graph.nodes[self.pnode_id].data["tokens"]  # n_paras * batch_size x n_tokens
        para_feature, context, _ = self.para_encoder(src)
        cluster_feature, _, __ = self.cluster_encoder(graph.nodes[self.cnode_id].data["id"])

        esedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)
        etf = graph.edges[esedge_id].data["ttfrac"]
        graph.edges[esedge_id].data["tfidfembed"] = self._TFembed(etf)

        # the start state
        cluster_state = cluster_feature
        para_state = self.cluster2para(graph, cluster_feature, para_feature)

        # TODO add gat_iter to args

        for i in range(self.args.gat_iter):
            cluster_state = self.para2cluster(graph, cluster_state, para_state)
            para_state = self.cluster2para(graph, cluster_state, para_state)

        para_state = self.layer_norm(para_state)
        para_state = para_state.unsqueeze(1)
        context = self.feed_forward(context + para_state)

        paras_in_one_batch, n_tokens = src.size()
        n_paras = 20
        assert paras_in_one_batch % n_paras == 0
        batch_size = paras_in_one_batch // n_paras

        mask_local = ~(src.data.eq(self.padding_idx).view(-1, n_tokens).bool())
        mask_hier = mask_local[:, :, None]  # n_paras * batch_size x n_tokens x 1
        context = context * mask_hier  # n_paras * batch_size x n_tokens x embed_dim
        context = context.view(batch_size, n_paras * n_tokens, -1)
        context = context.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim

        mask_hier = mask_hier.view(batch_size, n_paras * n_tokens, -1).bool()
        mask_hier = mask_hier.transpose(0, 1).contiguous()  # src_len, batch_size, 1

        unpadded = [torch.masked_select(context[:, i], mask_hier[:, i]).view([-1, context.size(-1)])
                    for i in range(
                context.size(1))]  # [tensor(src_len1 x embed_dim), tensor(src_len2 x embed_dim), ...] without pad token
        max_l = max([p.size(0) for p in unpadded])  # max_src_len
        mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).bool().to(self.device)
        mask_hier = ~mask_hier[:, None,
                     :]  # real_batch_size x 1 x max_src_len, result after concat all the paras in one example
        src_features = torch.stack(
            [torch.cat([p, torch.zeros(max_l - p.size(0), context.size(-1)).to(self.device)]) for p in unpadded],
            1)  # max_src_len x real_batch_size x embed_dim

        return src_features, mask_hier