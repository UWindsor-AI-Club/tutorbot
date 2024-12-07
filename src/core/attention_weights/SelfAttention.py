import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)


x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, dimensions=3
d_out = 2 # the output embedding size, d=2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

print(key_2)

keys = inputs @ W_key 
values = inputs @ W_value

print("keys:", keys)

#print("keys.shape:", keys.shape) # prints [6,2] 6 rows 2 columns
#print("values.shape:", values.shape)  # same thing


# finding siimmilarities (attention scores key dot query)

keys_2 = keys[1] # Python starts index at 0
attn_score_22 = query_2.dot(keys_2)
print("attention Score for Query 2, Key 2: ", attn_score_22)

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print("attention Scores for Query 2: ", attn_scores_2)

# Normalize attention scores to get weights
# this time we square root the denominator
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attention Weights for Query 2: ", attn_weights_2)

# now compute context vector 2
context_vec_2 = attn_weights_2 @ values
print("Context Vector for Query 2: ", context_vec_2)