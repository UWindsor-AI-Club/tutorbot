import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))


    def forward(self, input):
        keys = input @ self.W_key
        values = input @ self.W_value
        queries = input @ self.W_query

        attn_scores = queries @ keys.T # omega (unormalized attention scores)

        d_k = keys.shape[1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1) # alpha (attention weights sun attn scores / sqrt(dimenstion))

        context_vecs = attn_weights @ values
        return context_vecs


torch.manual_seed(123)

'''
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
'''
inputs = torch.tensor(
  [[0.43, 0.15, 0.69], # Your     (x^1)
   [0.55, 0.37, 0.66], # journey  (x^2)
   [0.55, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.18, 0.33], # with     (x^4)
   [0.77, 0.55, 0.90], # one      (x^5)
   [0.05, 0.50, 0.55]] # step     (x^6)
)


sa_v1 = SelfAttentionV1(d_in=3, d_out=2)
print(sa_v1(inputs))