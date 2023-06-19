import torch.nn as nn
from helpers.clip.core.clip import build_model, load_clip
import torch
from agents.l2a import L2A

class CommandClassifier(nn.Module):
    def __init__(self, input_size=512, device='cuda:1', l2a_weights=None):
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
        self.l2a = L2A(h1=1029).to(device)
        self.l2a.load_state_dict(torch.load(l2a_weights))

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
        self.sentence_emb = sentence_emb
        # pass the sentence embedding through the classifier
        out = self.forward(sentence_emb)
        return out