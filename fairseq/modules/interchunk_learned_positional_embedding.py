import torch.nn as nn
import torch
class InterchunkLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self,num_embedding:int,embedding_dim:int,padding_idx:int):
        super().__init__(num_embedding,embedding_dim,padding_idx)

    def forward(self,input):
        positions = self.make_positions(input)
        positions = positions.reshape(len(input),-1)

        return super().forward(positions)
    def make_positions(self,tensor):
        '''
            make inter chunk positions 
        '''
        mask = torch.ones_like(tensor)
        return torch.cumsum(mask, dim=1).long()





    
