import torch
from fairseq.models.transformer import (
    TransformerDecoder,
)
import torch.nn.functional as F
from fairseq.modules import InsidechunkLearnedPositionalEmbedding
from fairseq.modules import InterchunkLearnedPositionalEmbedding

import pdb
class TransformerSarDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens,no_encoder_attn) 
        # define chunk positional embedding
        self.inside_chunk_position_embedding = InsidechunkLearnedPositionalEmbedding(args.chunk_size + embed_tokens.padding_idx + 1, args.decoder_input_dim, embed_tokens.padding_idx)
        self.inter_chunk_position_embedding = InterchunkLearnedPositionalEmbedding(args.max_num_chunk, args.decoder_input_dim, embed_tokens.padding_idx)
    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, return_attn=False, src_attn_mask=None, incremental_update=True, **unused):
        """
        Similar to *forward* but only return features. Q：what features means

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # todo:masked matrix of semi-autoregresive
        self_attn_mask = self.buffered_future_mask(prev_output_tokens)
        self_attn_mask = self_attn_mask.cuda()

        # todo: inside chunk positioin embedding and inter chunkposition embedding
        inside_chunk_positions = self.inside_chunk_position_embedding(prev_output_tokens)
        inter_chunk_position = self.inter_chunk_position_embedding(prev_output_tokens)
        
        #todo: reshape the data: B X num_chunk X chunk_size --> B X T 
        prev_output_tokens = prev_output_tokens.reshape(prev_output_tokens.size(0),-1)
        # embed positions

        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None


        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        # add inter chunk position embedding & inside chunk position embedding
        x += inside_chunk_positions #
        x += inter_chunk_position
        
        # if positions is not None:
        #     x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn_lists = []

        inner_states = [x]

        # decoder layers
        for layer in self.layers:

            x, attn = layer( # bug在这里
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                incremental_update=incremental_update,
            )
            inner_states.append(x)
            attn_lists.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if return_attn:
            return x, {'attn': attn_lists, 'inner_states': inner_states} #
        else:
            return x, {'attn': None, 'inner_states': inner_states}

    def buffered_future_mask(self, tensor): # make noncausal mask matrix
        num_chunk = tensor.size(1)
        chunk_size = tensor.size(2)
        dim = num_chunk * chunk_size
        self._future_mask = torch.zeros(dim,dim) 
        for i in range(num_chunk):
            self._future_mask[chunk_size * (i-1): chunk_size * i, chunk_size * i:] = float('-inf')
        return self._future_mask[:dim, :dim] #