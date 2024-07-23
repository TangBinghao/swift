import torch
from torch import nn
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    def forward(self, hidden_state, attention_mask=None):
        # hidden_state bsz, seq_len, hidden_dim
        bsz = hidden_state.size(0)
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value) # bsz, num_heads, seq_len, head_dim

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(self.head_dim)

        if attention_mask != None:
            attention_scores += attention_mask * -1e9
        attention_probs = torch.softmax(attention_scores, dim=-1) # bsz, num_heads, seq_len, seq_len
        output = torch.matmul(attention_probs, value) # bsz, num_heads, seq_len, head_dim
        output = output.transpose(-1, -2).contiguous().view(bsz, -1, self.head_dim * self.num_heads)
        output = self.o_proj(output)
        return output
    def split_heads(self, x):
        bsz = x.size(0)
        return x.view(bsz, -1, self.num_heads, self.head_dim).transpose(1,2)


import torch
from torch import nn
class GroupQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * self.num_groups)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * self.num_groups)

        self.o_proj = nn.Linear(hidden_size, hidden_size)
    def forward(self, hidden_state, attention_mask=None):
        # hidden_state bsz, seq_len, hidden_dim
        bsz = hidden_state.size(0)
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        query = self.split_heads(query) # bsz, num_heads, seq_len, head_dim
        key = self.split_heads(key, self.num_groups)
        value = self.split_heads(value, self.num_groups) 

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(self.head_dim)

        if attention_mask != None:
            attention_scores += attention_mask * -1e9
        attention_probs = torch.softmax(attention_scores, dim=-1) # bsz, num_heads, seq_len, seq_len
        output = torch.matmul(attention_probs, value) # bsz, num_heads, seq_len, head_dim
        output = output.transpose(-1, -2).contiguous().view(bsz, -1, self.head_dim * self.num_heads)
        output = self.o_proj(output)
        return output
    def split_heads(self, x, num_groups=None):
        bsz, seq_len = x.size(0), x.size(1)
        if num_groups==None:
            return x.view(bsz, -1, self.num_heads, self.head_dim).transpose(1,2)
        else:
            # K,V [bsz, seq_len, group_num * head_dim]
            x = x.view(bsz, -1, self.num_groups, self.head_dim).transpose(1,2) # [bsz, group_num, seq_len,  head_dim]
            x = x[:,:,None,:,:].expand(bsz,self.num_groups, self.num_heads//self.num_groups, seq_len, self.head_dim).reshape(bsz, self.num_heads//self.num_groups*self.num_groups, seq_len, self.head_dim)
            return x 

import torch
from torch import nn
class MultiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups=1):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * self.num_groups)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * self.num_groups)

        self.o_proj = nn.Linear(hidden_size, hidden_size)
    def forward(self, hidden_state, attention_mask=None):
        # hidden_state bsz, seq_len, hidden_dim
        bsz = hidden_state.size(0)
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        query = self.split_heads(query) # bsz, num_heads, seq_len, head_dim
        key = self.split_heads(key, self.num_groups)
        value = self.split_heads(value, self.num_groups) 

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(self.head_dim)

        if attention_mask != None:
            attention_scores += attention_mask * -1e9
        attention_probs = torch.softmax(attention_scores, dim=-1) # bsz, num_heads, seq_len, seq_len
        output = torch.matmul(attention_probs, value) # bsz, num_heads, seq_len, head_dim
        output = output.transpose(-1, -2).contiguous().view(bsz, -1, self.head_dim * self.num_heads)
        output = self.o_proj(output)
        return output
    def split_heads(self, x, num_groups=None):
        bsz, seq_len = x.size(0), x.size(1)
        if num_groups==None:
            return x.view(bsz, -1, self.num_heads, self.head_dim).transpose(1,2)
        else:
            # K,V [bsz, seq_len, group_num * head_dim]
            x = x.view(bsz, -1, self.num_groups, self.head_dim).transpose(1,2) # [bsz, group_num, seq_len,  head_dim]
            x = x[:,:,None,:,:].expand(bsz,self.num_groups, self.num_heads//self.num_groups, seq_len, self.head_dim).reshape(bsz, self.num_heads//self.num_groups*self.num_groups, seq_len, self.head_dim)
            return x 