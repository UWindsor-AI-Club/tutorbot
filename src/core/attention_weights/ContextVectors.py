import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query

# get unormalized attention scores
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print("\nAttention scores unormalized for x2: ", attn_scores_2)


# normalize the attention scores we got
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights normalized for x2:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# multiply (dot product) the second token array with its' attention weight (from query2),
# resulting in a 1d array that is added up to get the context vector
query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print("\nContext Vector 2: ", context_vec_2)


# Attention Scores for All inputs (Ws)
print("\nAttention Scores for All Inputs:")

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# equivalent to : attn_scores = inputs @ inputs.T

print(attn_scores)

# noramlize Attention scores (alphas)
attn_weights = torch.softmax(attn_scores, dim=-1)

print("\nAttention Weights (Normalized Attention scores) for All Inputs:")
print(attn_weights)

# all_context_vecs = attn_weights @ inputs
# equiivalent to :

all_context_vecs = torch.zeros(inputs.shape)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        all_context_vecs[i] += attn_weights[i, j] * x_j

print("\nAll Context Vectors:")
print(all_context_vecs)



