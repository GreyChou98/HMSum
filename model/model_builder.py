from model.transformer_encoder import NewTransformerEncoder
from model.GAT import ESEGAT
from model.transformer_decoder import TransformerDecoder
from model.neural import PositionwiseFeedForward, sequence_mask
from model.optimizer import Optimizer
from model.HMEncoder import HMEncoder

import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    optim = Optimizer(
        args.optim, args.lr, args.max_grad_norm,
        beta1=args.beta1, beta2=args.beta2,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps, model_size=args.enc_hidden_size)


    if args.train_from != '':
        optim.optimizer.load_state_dict(checkpoint['optim'])
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    optim.set_parameters(list(model.named_parameters()))
    return optim


def get_generator(dec_hidden_size, vocab_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator



class Summarizer(nn.Module):
    def __init__(self, args, word_padding_idx, vocab_size, device, checkpoint=None):
        super(Summarizer, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.device = device

        # self._TFembed = nn.Embedding(50, self.args.emb_size)
        src_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        self.padding_idx = word_padding_idx

        if self.args.share_embeddings:
            tgt_embeddings.weight = src_embeddings.weight

        self.encoder = HMEncoder(self.args, self.device, src_embeddings, self.padding_idx)
        # self.para_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
        #                                        self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)
        # self.cluster_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
        #                                        self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)
        #
        # self.cluster2para = ESEGAT(in_dim=self.args.emb_size,
        #                            out_dim=self.args.emb_size,
        #                            num_heads=self.args.heads,
        #                            attn_drop_out=self.args.enc_dropout,
        #                            ffn_inner_hidden_size=self.args.ff_size,
        #                            ffn_drop_out=self.args.enc_dropout,
        #                            feat_embed_size=self.args.emb_size,
        #                            layerType="E2S")
        #
        # self.para2cluster = ESEGAT(in_dim=self.args.emb_size,
        #                            out_dim=self.args.emb_size,
        #                            num_heads=self.args.heads,
        #                            attn_drop_out=self.args.enc_dropout,
        #                            ffn_inner_hidden_size=self.args.ff_size,
        #                            ffn_drop_out=self.args.enc_dropout,
        #                            feat_embed_size=self.args.emb_size,
        #                            layerType="S2E")
        #
        # self.layer_norm1 = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        # self.feed_forward = PositionwiseFeedForward(self.args.enc_hidden_size, self.args.ff_size, self.args.enc_dropout)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.heads,
            d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, device)
        if self.args.share_decoder_embeddings:
            self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            # checkpoint['model']
            keys = list(checkpoint['model'].keys())
            for k in keys:
                if ('a_2' in k):
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
                if ('b_2' in k):
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])

            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)

    def forward(self, graph, tgt):
        tgt = tgt[:-1]

        # self.cnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        self.pnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        src = graph.nodes[self.pnode_id].data["tokens"]     # n_paras * batch_size x n_tokens
        # para_feature, context, _ = self.para_encoder(src)
        # cluster_feature, _, __ = self.cluster_encoder(graph.nodes[self.cnode_id].data["id"])
        #
        # esedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)
        # etf = graph.edges[esedge_id].data["ttfrac"]
        # graph.edges[esedge_id].data["tfidfembed"] = self._TFembed(etf)
        #
        # # the start state
        # cluster_state = cluster_feature
        # para_state = self.cluster2para(graph, cluster_feature, para_feature)
        #
        #
        #
        # for i in range(self.args.gat_iter):
        #     cluster_state = self.para2cluster(graph, cluster_state, para_state)
        #     para_state = self.cluster2para(graph, cluster_state, para_state)
        #
        # para_state = self.layer_norm1(para_state)
        # para_state = para_state.unsqueeze(1)
        # context = self.feed_forward(context + para_state)
        #
        paras_in_one_batch, n_tokens = src.size()
        n_paras = 20
        assert paras_in_one_batch % n_paras == 0
        batch_size = paras_in_one_batch // n_paras
        #
        # mask_local = ~(src.data.eq(self.padding_idx).view(-1, n_tokens).bool())
        # mask_hier = mask_local[:, :, None]    # n_paras * batch_size x n_tokens x 1
        # context = context * mask_hier    # n_paras * batch_size x n_tokens x embed_dim
        # context = context.view(batch_size, n_paras * n_tokens, -1)
        # context = context.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        #
        # mask_hier = mask_hier.view(batch_size, n_paras * n_tokens, -1).bool()
        # mask_hier = mask_hier.transpose(0, 1).contiguous()   # src_len, batch_size, 1
        #
        # unpadded = [torch.masked_select(context[:, i], mask_hier[:, i]).view([-1, context.size(-1)])
        #             for i in range(context.size(1))]    # [tensor(src_len1 x embed_dim), tensor(src_len2 x embed_dim), ...] without pad token
        # max_l = max([p.size(0) for p in unpadded])    # max_src_len
        # mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).bool().to(self.device)
        # mask_hier = ~mask_hier[:, None, :]    # real_batch_size x 1 x max_src_len, result after concat all the paras in one example
        # src_features = torch.stack(
        #     [torch.cat([p, torch.zeros(max_l - p.size(0), context.size(-1)).to(self.device)]) for p in unpadded],
        #     1)     # max_src_len x real_batch_size x embed_dim
        src_features, mask_hier = self.encoder(graph)
        src = src.view(batch_size, n_paras, n_tokens)    # be consistent with the init decoder state
        dec_state = self.decoder.init_decoder_state(src, src_features)     # src: num_paras_in_one_batch x max_length
        decoder_outputs = self.decoder(tgt, src_features, dec_state, memory_masks=mask_hier)

        return decoder_outputs

