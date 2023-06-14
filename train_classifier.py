from agents.command_classifier import CommandClassifier
import torch
import torch.utils.data as data
import numpy as np
import os
import clip
from helpers.clip.core.clip import build_model, load_clip, tokenize


class CommandDataset(data.Dataset):
    def __init__(self, commands0, commands1):
        self.commands = commands0 + commands1
        # labels are 0 for agnostic commands and 1 for commands with objects
        labels0 = np.zeros(len(commands0))
        labels1 = np.ones(len(commands1))
        self.labels = np.concatenate((labels0, labels1))

    def __getitem__(self, index):
        if self.labels[index] == 0:
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])
        return self.commands[index], label

    def __len__(self):
        return len(self.commands)

def generate_agnostic_commands():
    text1 = ['move ']
    text2 = ['a tiny bit ', 'a little ', '', 'a lot ']
    text3 = ['left ', 'right ', 'forward ', 'backward ', 'up ', 'down ', 
             'left and forward ', 'left and backward ', 'left and up ', 'left and down ',
             'right and forward ', 'right and backward ', 'right and up ', 'right and down ',
             'forward and up ', 'forward and down ', 'backward and up ', 'backward and down ']
    text4 = ['slowly', '', 'quickly']
    commands = []
    for t1 in text1:
        for t2 in text2:
            for t3 in text3:
                for t4 in text4:
                    commands.append(t1+t2+t3+t4)

    return commands

def generate_commands():
    # move above task
    text1 = ['move above the ', 'go above the ']
    text2 = ['red ', 'green ', 'blue ', '']
    text3 = ['cylinder', 'cuboid', 'sphere', 'cone', 'object']
    commands = []
    for t1 in text1:
        for t2 in text2:
            for t3 in text3:
                if t2 == '' and t3 == 'object':
                    continue
                commands.append(t1 + t2 + t3)

    return commands

def main():
    device = 'cuda:0'
    # generate all the possible agnostic commands
    commands0 = generate_agnostic_commands()
    commands1 = generate_commands()
    # convert commands to clip embeddings
    model, _ = load_clip('RN50', jit=False, device=device)
    clip_model = build_model(model.state_dict())
    clip_model.to(device)
    del model
    embeddings0 = []
    embeddings1 = []
    for command in commands0:
        tokens = tokenize([command]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        embeddings0.append(sentence_emb.detach().cpu().numpy())
    for command in commands1:
        tokens = tokenize([command]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        embeddings1.append(sentence_emb.detach().cpu().numpy())

    # create a dataset from the embeddings
    dataset = CommandDataset(embeddings0, embeddings1)
    # create a dataloader from the dataset
    train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # create a command classifier
    model = CommandClassifier(input_size=1024).to(device)
    # training loop
    epochs = 20
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    train_losses = []
    val_losses = []
    best_loss = 1e9
    interval = 50
    for epoch in range(epochs):
        # loop through data loader
        model.train()
        for i, (command, label) in enumerate(train_loader):
            # move to device
            command = command.float().to(device).squeeze()
            label = label.float().to(device)

            optimizer.zero_grad()
            # get predictions
            out = model.forward(command)
            # calculate loss
            loss = loss_fn(out, label)
            # backprop
            # loss.backward(retain_graph = True)
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())
        # print mean loss
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {np.mean(train_losses):.4f}')
        train_losses = []
        
        if (epoch + 1) % interval == 0:
            torch.save(model.state_dict(), os.path.join('./', f'model_{epoch}.pt'))

        # check accuracy
        model.eval()
        correct = 0
        for i, (command, label) in enumerate(val_loader):
            # move to device
            command = command.float().to(device).squeeze(0)
            label = label.float().to(device)

            # get predictions
            out = model.forward(command)
            # convert to pred
            pred = torch.argmax(out, dim=1).item()
            # convert label to gorund truth
            gt = torch.argmax(label, dim=1).item()
            # count correct
            correct += pred == gt
        # print accuracy
        print('Accuracy: ', correct / (i+1))


if __name__ == '__main__':
    main()