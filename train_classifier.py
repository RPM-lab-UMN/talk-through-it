from calendar import c
from agents.command_classifier import CommandClassifier
import torch
import torch.utils.data as data
import numpy as np
import os
from helpers.clip.core.clip import build_model, load_clip, tokenize
import pandas as pd
import pickle
from rlbench.const import colors

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
    # move front commands
    commands.append('move in front of the blue cup')
    commands.append('move in front of the dispenser')
    commands.append('move in front of the apple')
    commands.append('move in front of the fridge handle')

    # move above commands
    commands.append('move above the left edge of the pot')
    commands.append('move above the left edge of the left burner')
    commands.append('move above the left edge of the right burner')
    commands.append('move above the plate')

    # open drawer motions
    commands.append('move in front of the top handle')
    commands.append('move in front of the middle handle')
    commands.append('move in front of the bottom handle')

    # slide block motions
    text = ['move in front of the block',
            'move left of the block',
            'move right of the block',
            'move behind the block']
    commands.extend(text)

    # sweep to dustpan motions
    commands.append('move in front of the broom')

    # meat off grill motions
    text = ['move above the chicken',
            'move above the steak',]
    commands.extend(text)

    # turn tap motions
    text = ['move to the right tap',
            'move to the left tap']
    commands.extend(text)

    # put in drawer motions
    text = ['move above the block']
    commands.extend(text)

    # close jar motions
    text = ['move above the lid']
    for c in colors:
        text.append(f'move above the {c} jar')
    commands.extend(text)

    # drag stick motions
    text = ['move above the stick']

    # stack blocks motions
    text = ['move above the platform']
    for c in colors:
        text.append(f'move above the left {c} block')
        text.append(f'move above the right {c} block')
    commands.extend(text)

    # screw bulb motions
    text = ['move above the lamp']
    for c in colors:
        text.append(f'move above the {c} bulb')

    # put in safe motions
    text = ['move in front of the money',
            'move in front of the top shelf',
            'move in front of the middle shelf',
            'move in front of the bottom shelf']
    commands.extend(text)

    # place wine motions
    text = ['move in front of the bottle',
            'move in front of the near side of the rack',
            'move in front of the far side of the rack',
            'move in front of the middle of the rack']
    commands.extend(text)

    # put in cupboard motions
    text = ['move in front of the cupboard']
    groceries = [
        'crackers',
        'chocolate jello',
        'strawberry jello',
        'soup',
        'tuna',
        'spam',
        'coffee',
        'mustard',
        'sugar']
    for g in groceries:
        text.append(f'move above the {g}')
    commands.extend(text)

    # sort shape
    text = []
    shapes = [
        'cube',
        'cylinder',
        'prism',
        'star',
        'moon',
    ]
    for s in shapes:
        text.append(f'move above the {s}')
        text.append(f'move above the {s} hole')
    commands.extend(text)

    # push buttons
    text = []
    for c in colors:
        text.append(f'move above the {c} button')
    commands.extend(text)

    # insert peg
    text = ['move above the square ring']
    for c in colors:
        text.append(f'move above the {c} peg')
    commands.extend(text)

    # stack cups
    text = []
    for c in colors:
        text.append(f'move above the {c} cup')
        text.append(f'move a lot above the {c} cup')
    commands.extend(text)

    # place cups
    text = ['move above the left cup',
            'move left of the rack']
    commands.extend(text)

    return commands

def main():
    device = 'cuda:0'
    # load the agnostic commands from csv file
    csv_path = os.path.join(os.getcwd(), 'commands.csv')
    df = pd.read_csv(csv_path)
    commands0 = df['command'].tolist()
    commands0.append('move back')
    commands0.append('keep moving')
    # generate object commands
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

    # all embeddings in one array
    embeddings = np.concatenate((embeddings0, embeddings1), axis=0).squeeze()
    # save embeddings to npy
    np.save('embeddings.npy', embeddings)

    # create a dataset from the embeddings
    dataset = CommandDataset(embeddings0, embeddings1)
    # create a dataloader from the dataset
    train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # create a command classifier
    cwd = os.path.dirname(os.path.realpath(__file__))
    l2a_path = os.path.join(cwd, 'l2a.pt')
    model = CommandClassifier(input_size=1024, l2a_weights=l2a_path).to(device)
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
        accuracy = correct / (i+1)
        print('Accuracy: ', accuracy)
        if accuracy == 1:
            torch.save(model.state_dict(), os.path.join('./', f'text_classifier.pt'))

if __name__ == '__main__':
    main()