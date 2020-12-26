import math

import torch.nn as nn
import torch

from model.attn import MultiHeadedAttention, MultiHeadedPooling
from model.neural import PositionwiseFeedForward, PositionalEncoding, sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        context = self.pooling_attn(inputs, inputs,
                                    mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)


class NewTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, device):
        super(NewTransformerEncoder, self).__init__()
        self.device = device
        self.d_model = d_model
        self.heads = heads
        self.d_per_head = self.d_model // self.heads
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = ~(src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens).bool())
        mask_block = (torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0).bool()

        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            word_vec = self.transformer_local[i](word_vec, word_vec, ~mask_local)  # all_sents * max_tokens * dim
        word_vec = self.layer_norm1(word_vec)
        mask_inter = (~mask_block).unsqueeze(1).expand(batch_size, self.heads, n_blocks).contiguous()
        mask_inter = mask_inter.view(batch_size * self.heads, 1, n_blocks).bool()
        block_vec = self.pooling(word_vec, word_vec, ~mask_local)
        block_vec = block_vec.view(-1, self.d_per_head)
        block_vec = self.layer_norm2(block_vec)
        block_vec = self.dropout2(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        src_features = self.feed_forward(word_vec + block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, -1)

        mask_hier = mask_local[:, :, None].float()
        src_features = src_features * mask_hier
        src_features = src_features.view(batch_size, n_blocks, n_tokens, -1)

        return block_vec, src_features, mask_hier

class NewTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(NewTransformerEncoderLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        :param inputs: [ num_of_paras_in_one_batch x seq_len x d_model]
        :param mask: [ num_of_paras_in_one_batch x seq_len ]
        :return:
        """
        batch_size, seq_len, d_model = inputs.size()
        input_norm = self.layer_norm(inputs)
        mask_local = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask_local)
        para_vec = self.dropout(context) + inputs
        para_vec = self.pooling(para_vec, para_vec, mask)
        para_vec = self.layer_norm2(para_vec)
        para_vec = para_vec.view(batch_size, -1)
        return para_vec


