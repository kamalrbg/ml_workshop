import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

base_model = 'bert-base-uncased'

b_model = AutoModel.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

max_input_length = 10
init_token_idx = 101
eos_token_idx = 102


class SentimentAnalyzer(nn.Module):
    def __init__(self,
                 b_model,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.b_model = b_model
        
        embedding_dim = b_model.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.b_model(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output
    
    
hidden_dim = 256
op_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.25

model = SentimentAnalyzer(b_model,
                         hidden_dim,
                         op_dim,
                         n_layers,
                         bidirectional,
                         dropout)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('dev-model.pt'))
model.to(device)

def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()





sentiment_analyser = FastAPI()

origins = ["*"]

sentiment_analyser.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@sentiment_analyser.post("/")
async def get_sentiments(text: str):
    prediction = predict_sentiment(model, tokenizer, text)
    if prediction >= 0.5:
        response = {"result": "postive"}
    else:
        response = {"result": "negative"}
    return response

if __name__ == "__main__":
    uvicorn.run("service:sentiment_analyser", host="0.0.0.0", port=9001)

