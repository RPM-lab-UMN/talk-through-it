import os
# add parent to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from helpers.clip.core.clip import build_model, load_clip
import torch
from agents.l2a import L2A
import numpy as np
from clip import tokenize

class CommandClassifier(nn.Module):
    def __init__(self, l2a_weights, input_size=512, device='cuda:1'):
        self.device = device
        super(CommandClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

        # create the clip model
        model, _ = load_clip('RN50', jit=False, device=device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(device)
        del model

        # keep last sentence embedding
        self.sentence_emb = None

        # load the l2a weights
        self.l2a = L2A(h1=1031).to(device)
        self.l2a.load_state_dict(torch.load(l2a_weights))

        # load the embeddings
        self.train_embeddings = torch.tensor(np.load('embeddings.npy')).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        out = self.softmax(x)
        return out
    
    def predict(self, tokens):
        # make the tokens an embedding with the clip model
        token_tensor = torch.from_numpy(tokens).to(self.device)
        sentence_emb, token_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
        self.sentence_emb = sentence_emb.float()
        # pass the sentence embedding through the classifier
        out = self.forward(self.sentence_emb)
        # change to 0 or 1 scalar value
        out = torch.argmax(out, dim=1)
        return out
    
    def similarity(self, embedding):
        # find the largest cosine similarty between the embedding and the train embeddings
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(embedding, self.train_embeddings)
        return torch.max(similarities)

if __name__ == "__main__":
    device = 'cuda:0'
    # create a command classifier
    cwd = os.path.dirname(os.path.realpath(__file__))
    l2a_path = os.path.join(cwd, '../l2a.pt')
    model = CommandClassifier(input_size=1024, l2a_weights=l2a_path, device=device).to(device)
    command = 'go above the plate'
    tokens = tokenize([command]).numpy()
    token_tensor = torch.from_numpy(tokens).to(model.device)
    sentence_emb, token_embs = model.clip_model.encode_text_with_embeddings(token_tensor)
    sentence_emb = sentence_emb.float()
    print(model.similarity(sentence_emb)) # say 0.95 is threshold
