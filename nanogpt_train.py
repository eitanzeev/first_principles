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
num_blocks = 6
num_heads = 6
attn_dropout = 0.2
residual_cxn_dropout = 0.2


#---

#TRain a transformer

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt 

#Get the input text
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    
#What is the length of the dataset in characters
#print(f"length of the dataset in characters: {len(text)}")


##Sort the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
all_chars = ''.join(chars)

print(f"All Chars: {all_chars}")
print(f"Total number: {vocab_size} chars")


#Lets work on tokenization

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [itos[c] for c in l]


data = torch.tensor(encode(text), dtype = torch.long)
#print("Every integer in data represents a single character ", data.shape)


#Train and Test Splits
n = int(0.9*len(data))

train_data = data[:n]
val_data = data[n:]


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
    def __init__(head_size, n_embd):
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
        
        q,k,v = kqv(x).split(n_embd, 2)

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
    
class MLP(nn.Module()):
    def __init__(self, n_embd):
        super().__init__()
        
        upscale = nn.Linear(n_embd, n_embd * 4, bias = False)
        nonlinear = nn.GELU()
        downscale = nn.Linear(n_embd*4, bias = False)
        dropout = nn.Dropout(general_dropout)
        
        self.sequential_MLP = nn.Sequential(upscale, nonlinear, downscale, dropout)
        
    def forward(self, x):
        return self.sequential_MLP(x)
        
        
        
        
#Adding the head class
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)#We don't typically use bias here
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.head_size = head_size
        
        #creating a variable tril for the model as it is not a inherent parameter for the model
        self.register_buffer('tril',torch.tril(torch.ones(context_length, context_length)))
        
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        v = self.value(x)
        
        #Calculate the affinities
        weights = q@k.transpose(-2,-1) *self.head_size*0.5 #(B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #Only look backwards: (B,T,T)
        weights = F.softmax(weights, dim = -1) # (B,T,T)    
        
        weights = self.dropout(weights)   
        
        #perform the weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = weights@v #(B,T,T)@(B,T,C) -> (B,T,C)
        return out        
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self,  num_heads, head_size):
        super().__init__()
        #Create a list of the heads of attention
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #Calculate the heads of attention and then concatenate along the last dimension - the channel dimension
        mlt_head = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.proj(mlt_head) #projection back into the residual layer
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    
    #Linear layer followed by non=linear
    
    def __init__(self, n_embd):
        super().__init__()
        
        
        """Note the up projection in feedforward layers with the projection component!
        This just directly mimics what we saw from the Attention is All you Need paper
        """
               
        ##Non-separated projection layer
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd,  4*n_embd, bias = False),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
            nn.Dropout(dropout)
            )
        
    def forward(self, x):
        
        ##Separated projection layer
        ffwd = self.feedforward(x)
        out = self.proj(ffwd)
        out = self.dropout(out)
        
        ##Non-separated projection layer
        #ffwd = self.feedforward(x)
        return out
    
class Block(nn.Module):
    
    def __init__(self,n_embd, num_heads):
        super().__init__()
        
        head_size = n_embd//num_heads
        
        self.multihead = CasualSelfAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
       
        #Apply layer normalization PRIOR to attention and feedforward
        x = x + self.multihead(self.ln1(x)) #add residual connections to preserve gradient flows in case multi-head or feedforward fail
        x = x + self.ffwd(self.ln2(x))  #add residual connections to preserve gradient flows in case multi-head or feedforward fail
        return x
    
class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # layer mean (note that we are summing across the row not per column like with batch)
    xvar = x.var(1, keepdim=True) # layer mean( note that we normalizing)
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
        
#Modle structure
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #Create embedding table where each index(representing individual characters) 
        # will pull out a row from the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #n_embd is number of embeddings per token
        #Create embedding table for position encodings
        self.position_embedding_table = nn.Embedding(context_length, n_embd) #we encode each position in a n_embd dimension vector
        
        # self.attention_blocks = nn.Sequential(Block(n_embd, num_heads = 4),
        #                             Block(n_embd, num_heads = 4),
        #                             Block(n_embd, num_heads = 4))
        self.attention_blocks = nn.Sequential(*[Block(n_embd, num_heads=num_heads) for i in range(num_blocks)])

        ##feedforward
        #self.feedforward = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) #linear layer taking embeddings to logits - serves as the decoding layer
        
        
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