import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

#hyperparameters section
batch_size = 32 #How many independent sequences of characters will we process in parallel
context_length = 8 #-Cody's version of block size. What is the maximum context length for predictions
max_iters = 3000
eval_interval = 12
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

print(f"{all_chars}")
print(f"{vocab_size} chars")


#Lets work on tokenization

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decoder = lambda l: [itos[c] for c in l]


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
    
    return x,y

#Modle structure
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        
        #Create embedding table where each index(representing individual characters) 
        # will pull out a row from the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
        
    def forward(self, idx, targets = None):
       
        #idx and targets are both (b, context_length) tensor of integers
        logits = self.token_embedding_table(idx) #output is (B,T,C)
        
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
            logits, loss = self(idx)
            
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

m = BigramLanguageModel(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(params = m.parameters(), lr = learning_rate)


for _ in max_iters:
    
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
    
    
    
    
xb, yb = get_batch('train')
