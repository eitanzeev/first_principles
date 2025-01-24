import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

#hyperparameters section
batch_size = 32 #How many independent sequences of characters will we process in parallel
context_length = 8 #-Cody's version of block size. What is the maximum context length for predictions
max_iters = 3000 #500#3000
eval_interval = 12
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

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


#Adding the head class
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)#We don't typically use bias here
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.head_size = head_size
        
        #creating a variable tril for the model
        self.register_buffer('tril',torch.tril(torch.ones(context_length, context_length)))
        
    def forward(self, x):
        B,T,C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        weights = q@k.transpose(-2,-1) *self.head_size*80.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim = -1)        
        
        v = self.value(x)
        out = weights@v
        return out        
        
#Modle structure
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #Create embedding table where each index(representing individual characters) 
        # will pull out a row from the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #n_embd is number of embeddings per token
        
        #Create embedding table for position encodings
        self.position_embedding_table = nn.Embedding(context_length, n_embd) #we encode each position in a n_embd dimension vector
        self.sa_head = Head(n_embd) #define a single head of attention
        self.lm_head = nn.Linear(n_embd, vocab_size) #linear layer taking embeddings to logits
        
        
    def forward(self, idx, targets = None):
        
        B, T = idx.shape
       
        #idx and targets are both (b, context_length) tensor of integers
        #Create embedding table
        token_embeddings = self.token_embedding_table(idx) #output is (B,T,C = n_embd)
        
        #Create embedding table for position encodings, each of T in context length is encoded to n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #T,C structure
        
        #Combine the impact of the token and position embeddings
        comb_embd = token_embeddings + pos_emb #new dimension added with broadcasting
        attention_result = self.sa_head(comb_embd) #aplpy one head of attention
        logits = self.lm_head(comb_embd) #B,T,vocab_size)
        
        
        
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
            
            idx_cond= idx[:, -context_length]
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