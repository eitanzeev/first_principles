import tiktoken

class TokenizationProcedure():
    def __init__(self, processing_type, **character_lvl_args):
        self.processing_type = processing_type
        
        if self.processing_type == 'byte':
            self.tokenizer = tiktoken.get_encoding("o200k_base") #is it valuable to extend this depending on the tokenizer?
        elif self.processing_type == 'bigram':
            print("WARNING: Using standard Bigram tokenizer")
        else:
            raise TypeError("Incorrect processing type provided. Please check")
    
    def encode_text(self, text):
    
        if self.processing_type == 'byte':   
            tokens = self.tokenizer.encode(text)
        
        
        elif self.processing_type == 'bigram':
            ##Sort the text
            self.chars = sorted(list(set(text)))
            vocab_size = len(self.chars)
            all_chars = ''.join(self.chars)

            print(f"All Chars: {all_chars}")
            print(f"Total number: {vocab_size} chars")

            #Lets work on tokenization
            stoi = {ch:i for i,ch in enumerate(self.chars)}
            encode = lambda s: [stoi[c] for c in s]
            tokens = encode(text)
            


        
        return tokens
    
    def decode_text(self, tokens):
        if self.processing_type == 'byte':   
            text = self.tokenizer.decode(tokens)
            
        elif self.processing_type == 'bigram':
            itos = {i:ch for i,ch in enumerate(self.chars)}
            decode = lambda l: [itos[c] for c in l]
            text = ''.join(decode(tokens))
            
            
        
        return text
    