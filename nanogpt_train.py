import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

#hyperparameters section
batch_size = 64 #How many independent sequences of characters will we process in parallel
context_length = 256 #-Cody's version of block size. What is the maximum context length for predictions
max_iters = 3500#3000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
gen_dropout = 0.2
num_blocks = 2
num_heads = 6
attn_dropout = 0.2
residual_cxn_dropout = 0.2


#---

#TRain a transformer

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt 


#Batch splitting
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,)) #tensor of four integers representing four characters from the text
    # So for the random characters, we get i and then up to the block size. Then we put in a bigger tensor
    x = torch.stack([data[i:i+context_length] for i in ix])
    #same thing for y
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    
    #for gpu compatibility
    x = x.to(device)
    y= y.to(device)
    
    return x,y

@torch.no_grad()#Everything that happens inside this function, don't keep track of grad on estimate loss
def estimate_loss():
    
    """This function is used for switching between eval and train model to routinely
    evaluate model performance"""
    
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    #Switch back to eval 
    model.train()
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, head_size, n_embd, dropout):
        
        super().__init__()
        
        kqv = nn.Linear(n_embd, n_embd*3, bias = False)
        #creating a variable tril for the model as it is not a inherent parameter for the model
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length))\
                                    .view(1,1,context_length, context_length))
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual_dropout = nn.Dropout(residual_cxn_dropout)
        
        self.projection = nn.Linear(n_embd, n_embd)
        
    
    def forward(self, x):
        
        B,T,C = x.shape
        
        q,k,v = self.kqv(x).split(n_embd, 2)

        #adjust queries to have the head dimension 
        k = k.view(B,T, num_heads,C//num_heads).transpose(1,2) #B,T,C -> B,T,nH, C -> B,nH,T,C
        q = q.view(B,T,num_heads,C//num_heads).transpose(1,2)
        v = v.view(B,T,num_heads, C//num_heads).transpose(1,2)
        
        attention = (q@k.transpose(-1,-2))*(head_size**0.5)
        
        attention = attention.masked_fill(tril[:,:,:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim = -1)
        
        #Apply dropout
        attention = attn_dropout(attention)
        
        
        #value multiplication to 'catch' the attention weights
        #B,nH, T,T -> B, nH, T, C
        output = attention@v 
        
        #turn the data into continuous stream and effectively concatenate all of the heads again
        output = output.transpose(1,2).reshape(B,T,C)
        
        output = self.residual_dropout(self.projection(output))
        
        return output
    
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        
               
        """Note the up projection in feedforward layers with the projection component!
        This just directly mimics what we saw from the Attention is All you Need paper
        """
        
        upscale = nn.Linear(n_embd, n_embd * 4, bias = False)
        nonlinear = nn.GELU()
        downscale = nn.Linear(n_embd*4, bias = False)
        dropout = nn.Dropout(gen_dropout)
        
        self.sequential_MLP = nn.Sequential(upscale, nonlinear, downscale, dropout)
        
    def forward(self, x):
        return self.sequential_MLP(x)
            
class Block(nn.Module):
    
    def __init__(self,n_embd, num_heads):
        super().__init__()
        
        head_size = n_embd//num_heads
        
        self.multihead_attention = CausalSelfAttention(num_heads, head_size)
        self.ln1 = LayerNorm(n_embd)
        self.MLP = MLP(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
    def forward(self, x):
       
        #Apply layer normalization PRIOR to attention and feedforward
        x = x + self.multihead_attention(self.ln1(x)) #add residual connections to preserve gradient flows in case multi-head or feedforward fail
        x = x + self.MLP(self.ln2(x))  #add residual connections to preserve gradient flows in case multi-head or feedforward fail
        return x
    
class LayerNorm(nn.Module): # (used to be BatchNorm1d)

    def __init__(self, ndim, eps=1e-5, momentum=0.1, bias = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim) if bias else None) #bias is normally nothing
        
    def forward(self, x):
        #we normalize according to layer shape, which is represented by the weight.shape.
        #each parameter here represents a learned representation
        normed_output = F.layer_norm(x, self.weight.shape, self.weight, self.bias) 
        
        return normed_output

#Modle structure
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #Create embedding table where each index(representing individual characters) 
        # will pull out a row from the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #n_embd is number of embeddings per token
        #Create embedding table for position encodings
        self.position_embedding_table = nn.Embedding(context_length, n_embd) #we encode each position in a n_embd dimension vector
        self.attention_blocks = nn.Sequential(*[Block(n_embd, num_heads=num_heads) for i in range(num_blocks)])
        #linear layer taking embeddings to logits - serves as the decoding layer
        
        self.lm_head = nn.Linear(n_embd, vocab_size) 
        
    def forward(self, idx, targets = None):
        
        B, T = idx.shape
       
        #idx and targets are both (b, context_length) tensor of integers
        #Create embedding table
        token_embeddings = self.token_embedding_table(idx) #output is (B,T,C = n_embd)
        
        #Create embedding table for position encodings, each of T in context length is encoded to n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #T,C structure
        
        #Combine the impact of the token and position embeddings
        comb_embd = token_embeddings + pos_emb #new dimension added with broadcasting
        
        
        #attention_result = self.sa_heads(comb_embd) #apply one instance of attention
        #use blocks of attention
        attention_block_result = self.attention_blocks(comb_embd)
        
        logits = self.lm_head(attention_block_result) #B,T,vocab_size)
        
        if targets is None: 
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) #-> for cross entropy
            targets = targets.view(B*T) #reshaping
            
            #negative log likelihood
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        
        #idx is (B,context_length) array of indices in the current context. Predicting token by token
        for _ in range(max_new_tokens):
            
            
            """Crop the index to the last context_length of tokens. 
            We can't have more than context length coming in from idx 
            otherwise the embedding table will known out of scope"""
            idx_cond= idx[:, -context_length:]
            logits, loss = self(idx_cond)
            
            #focus only on the last example (where we consider the max token length TODO: Understand why here)
            #These are the predictions for what comes next
            logits = logits[:,-1,:]
            
            #Apply softmax to get the probabilities
            probs = F.softmax(logits, dim = -1) #B, C
            
            idx_next = torch.multinomial(probs, num_samples = 1) #for each context length in the batch return the next selection based on probabilities distribution
            
            idx = torch.cat((idx, idx_next), dim = 1) #concatenate to the idx to create (B, T+1)
            #print(f"current status of idx: {idx}")
            
        return idx


##Run the model
#Initiate the model

model = BigramLanguageModel()
m = model.to(device)

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(params = m.parameters(), lr = learning_rate)


for _ in range(max_iters):
    
    #evaluate loss
        if _ % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {_}: train loss {losses['train']:4f}, val loss {losses['val']:.4f}")
        
        #Get the batches
        xb, yb = get_batch('train')
        
        #Run the forward pass
        logits, loss = m(xb, yb)
        
        #zero the gradients prior to running gradient calculations
        optimizer.zero_grad(set_to_none=True)
        
        #Calculate the gradients and use them to update the parameters
        loss.backward()
        
        #step the optimizer to update the parameters
        optimizer.step()
    
    
    
#Generation section
# starting_index = torch.zeros((1,1), dtype = torch.long, device = device) #Kicking off generation with zero tensors
# sample_generated = m.generate(starting_index, max_new_tokens=100)
# print(f"sample generated: {sample_generated.shape}")
# #Sample generated works on the order of batches so we need to grab the return batch (it's effectively nested in a list)
# batch_of_interest = sample_generated[0]

#Turn it into a list and decode
#batch_of_interest = [str(x) for x in batch_of_interest]
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(''.join(decode(m.generate(context, max_new_tokens = 500)[0].tolist())))
# decoded = decoder(''.join(batch_of_interest.tolist()))
# print(f"Post training result: {decoded}")
