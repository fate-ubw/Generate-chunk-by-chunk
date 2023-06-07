import torch.nn as nn
import torch
class InsidechunkLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self,num_embedding:int,embedding_dim:int,padding_idx:int):
        super().__init__(num_embedding,embedding_dim,padding_idx)

    def forward(self,input):
        positions = self.make_positions(input,self.padding_idx)
        positions = positions.reshape(len(input),-1)

        return super().forward(positions)

    def make_positions(self,tensor, padding_idx):
        '''
            make inter chunk positions 
        '''
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=2).type_as(mask) * mask).long() + padding_idx



    
