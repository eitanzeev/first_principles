import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


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

class CausalSelfAttention(nn.Module):
    def __init__(self, gptconfig):
        
        super().__init__()
        
        
        self.kqv = nn.Linear(gptconfig.n_embd, gptconfig.n_embd*3, bias = False)
        #creating a variable tril for the model as it is not a inherent parameter for the model
        self.register_buffer('tril', torch.tril(torch.ones(gptconfig.context_length, gptconfig.context_length))\
                                    .view(1,1, gptconfig.context_length, gptconfig.context_length))
        self.dropout = nn.Dropout(gptconfig.gen_dropout)
        self.attn_dropout = nn.Dropout(gptconfig.attn_dropout)
        self.residual_dropout = nn.Dropout(gptconfig.residual_cxn_dropout)
        self.projection = nn.Linear(gptconfig.n_embd, gptconfig.n_embd)
        
        #To pass to forward
        self.num_heads = gptconfig.num_heads
        self.n_embd = gptconfig.n_embd
        self.head_size =gptconfig.head_size
        
    
    def forward(self, x):
        
        B,T,C = x.shape
        
        q,k,v = self.kqv(x).split(self.n_embd, 2)

        #adjust queries to have the head dimension 
        k = k.view(B,T, self.num_heads,C//self.num_heads).transpose(1,2) #B,T,C -> B,T,nH, C -> B,nH,T,C
        q = q.view(B,T,self.num_heads,C//self.num_heads).transpose(1,2)
        v = v.view(B,T,self.num_heads, C//self.num_heads).transpose(1,2)
        
        attention = (q@k.transpose(-1,-2))*(self.head_size**0.5)
        
        attention = attention.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim = -1)
        
        #Apply dropout
        attention = self.attn_dropout(attention)
        
        
        #value multiplication to 'catch' the attention weights
        #B,nH, T,T -> B, nH, T, C
        output = attention@v 
        
        #turn the data into continuous stream and effectively concatenate all of the heads again
        output = output.transpose(1,2).reshape(B,T,C)
        
        output = self.residual_dropout(self.projection(output))
        
        return output
    
class MLP(nn.Module):
    def __init__(self, gptconfig):
        super().__init__()
        
               
        """Note the up projection in feedforward layers with the projection component!
        This just directly mimics what we saw from the Attention is All you Need paper
        """
        
        upscale = nn.Linear(gptconfig.n_embd, gptconfig.n_embd * 4, bias = False)
        nonlinear = nn.GELU()
        downscale = nn.Linear(gptconfig.n_embd*4, gptconfig.n_embd, bias = False)
        dropout = nn.Dropout(gptconfig.gen_dropout)
        
        self.sequential_MLP = nn.Sequential(upscale, nonlinear, downscale, dropout)
        
    def forward(self, x):
        return self.sequential_MLP(x)
            
class Block(nn.Module):
    
    def __init__(self, gptconfig):
        super().__init__()
        
        
        self.multihead_attention = CausalSelfAttention(gptconfig)
        self.ln1 = LayerNorm(gptconfig.n_embd)
        self.MLP = MLP(gptconfig)
        self.ln2 = LayerNorm(gptconfig.n_embd)
        
    def forward(self, x):
       
        #Apply layer normalization PRIOR to attention and feedforward
        x = x + self.multihead_attention(self.ln1(x)) #add residual connections to preserve gradient flows in case multi-head or feedforward fail
        x = x + self.MLP(self.ln2(x))  #add residual connections to preserve gradient flows in case multi-head or feedforward fail
        return x
    
class LayerNorm(nn.Module): # (used to be BatchNorm1d)

    def __init__(self, ndim, eps=1e-5, momentum=0.1, bias = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None #bias is normally nothing
        
    def forward(self, x):
        #we normalize according to layer shape, which is represented by the weight.shape.
        #each parameter here represents a learned representation
 
        normed_output = F.layer_norm(x, self.weight.shape, self.weight, self.bias if self.bias is not None else None) 
        return normed_output



class GPTConfig:
    
    def __init__(self, **config):
        # Hyperparameters section
        self.batch_size = config.get('batch_size', 64)  # Parallel processing sequences
        self.context_length = config.get('context_length', 256)  # Max context length
        self.max_iters = config.get('max_iters', 3500)  # Number of training iterations
        self.eval_interval = config.get('eval_interval', 500)  # Evaluation frequency
        self.learning_rate = config.get('learning_rate', 3e-4)  # Learning rate
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')  # Auto-detect device
        self.eval_iters = config.get('eval_iters', 200)  # Iterations for evaluation
        self.n_embd = config.get('n_embd', 384)  # Embedding dimension
        self.gen_dropout = config.get('gen_dropout', 0.2)  # General dropout rate
        self.num_blocks = config.get('num_blocks', 2)  # Number of transformer blocks
        self.num_heads = config.get('num_heads', 6)  # Attention heads per block
        self.attn_dropout = config.get('attn_dropout', 0.2)  # Attention dropout rate
        self.residual_cxn_dropout = config.get('residual_cxn_dropout', 0.2)  # Dropout in residual connections
        self.vocab_size = config.get('vocab_size', 65)
        self.head_size = config.get('head_size',64)
        
#Modle structure
class PrelimGPT(nn.Module):
    
    def __init__(self, gptconfig):
        super().__init__()
        
        
        #Create embedding table where each index(representing individual characters) 
        # will pull out a row from the embedding table
        self.token_embedding_table = nn.Embedding(gptconfig.vocab_size, gptconfig.n_embd) #n_embd is number of embeddings per token
        #Create embedding table for position encodings
        self.position_embedding_table = nn.Embedding(gptconfig.context_length, gptconfig.n_embd) #we encode each position in a n_embd dimension vector
        self.attention_blocks = nn.Sequential(*[Block(gptconfig) for i in range(gptconfig.num_blocks)])
        
        #Create dictionary for attention blocks
        self.full_transformer = nn.ModuleDict(dict(token_embedding = self.token_embedding_table, 
                                                position_embedding = self.position_embedding_table,
                                                attention_blocks = self.attention_blocks))
        
        
        #linear layer taking embeddings to logits - serves as the decoding layer
        self.lm_head = nn.Linear(gptconfig.n_embd, gptconfig.vocab_size) 
        
        #Create all layers for easy structure
        self.all_layers = nn.Sequential(self.full_transformer, self.lm_head)
        
    def forward(self, idx, targets = None):
        
        B, T = idx.shape
        pos = torch.arange(T, device = device)
        #idx and targets are both (b, context_length) tensor of integers
        
        #Create embedding table
        token_embeddings = self.full_transformer.token_embedding(idx) #output is (B,T,C = n_embd)
        #Create embedding table for position encodings, each of T in context length is encoded to n_embd
        pos_emb = self.full_transformer.position_embedding(pos) #T,C structure
        
        #Combine the impact of the token and position embeddings
        comb_embd = token_embeddings + pos_emb #new dimension added with broadcasting
        
        #attention_result = self.sa_heads(comb_embd) #apply one instance of attention
        #use blocks of attention
        attention_block_result = self.full_transformer.attention_blocks(comb_embd)
        
        #Linear layer projection to logits
        logits = self.lm_head(attention_block_result) #B,T,vocab_size)
        
        if targets is None: 
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) #-> for cross entropy
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



if __name__ == "__main__":
    pass
    # ##Run the model
    # #Initiate the model

    # model = BigramLanguageModel()
    # m = model.to(device)

    # #create a PyTorch optimizer
    # optimizer = torch.optim.AdamW(params = m.parameters(), lr = learning_rate)


    # for _ in range(max_iters):
        
    #     #evaluate loss
    #         if _ % eval_interval == 0:
    #             losses = estimate_loss()
    #             print(f"step {_}: train loss {losses['train']:4f}, val loss {losses['val']:.4f}")
            
    #         #Get the batches
    #         xb, yb = get_batch('train')
            
    #         #Run the forward pass
    #         logits, loss = m(xb, yb)
            
    #         #zero the gradients prior to running gradient calculations
    #         optimizer.zero_grad(set_to_none=True)
            
    #         #Calculate the gradients and use them to update the parameters
    #         loss.backward()
            
    #         #step the optimizer to update the parameters
    #         optimizer.step()
        
        
        
    # #Generation section
    # # starting_index = torch.zeros((1,1), dtype = torch.long, device = device) #Kicking off generation with zero tensors
    # # sample_generated = m.generate(starting_index, max_new_tokens=100)
    # # print(f"sample generated: {sample_generated.shape}")
    # # #Sample generated works on the order of batches so we need to grab the return batch (it's effectively nested in a list)
    # # batch_of_interest = sample_generated[0]

    # #Turn it into a list and decode
    # #batch_of_interest = [str(x) for x in batch_of_interest]
    # context = torch.zeros((1,1), dtype = torch.long, device = device)
    # print(''.join(decode(m.generate(context, max_new_tokens = 500)[0].tolist())))
    # # decoded = decoder(''.join(batch_of_interest.tolist()))
    # # print(f"Post training result: {decoded}")
