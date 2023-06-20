import numpy as np
import torch
import clip
import sys
import os
from agents.l2a import L2A
import matplotlib.pyplot as plt
import torch.utils.data as data
from helpers.clip.core.clip import build_model, load_clip

class LangDataset(data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def get_dataset(values, directions, speeds, clip_model):
    device = 'cuda:1'
    i = 0
    commands = []
    labels = []
    dist = 10
    for d in directions:
        for v in values:
            for s in speeds:
                # generate command
                command = 'move ' + v + d + s
                # generate the label
                label = np.array([0, 0, 0, 0, 0], dtype=np.float32)
                if 'left' in d:
                    label[0] = -dist
                elif 'right' in d:
                    label[0] = dist
                if 'forward' in d:
                    label[1] = dist
                elif 'backward' in d:
                    label[1] = -dist
                if 'up' in d:
                    label[2] = dist
                elif 'down' in d:
                    label[2] = -dist
                # modify label based on value
                if 'a little' in v:
                    label = label / 2
                elif 'a lot' in v:
                    label = label * 2
                elif 'a tiny bit' in v:
                    label = label / 10
                # divide by sqrt 2 if two directions
                if 'and' in d:
                    label = label / 2**0.5
                # set the speed label
                label[3] = dist
                if 'slowly' in s:
                    label[3] = dist / 2
                elif 'quickly' in s:
                    label[3] = dist * 2
                # alternate gripper label
                if i % 2 == 0:
                    label[4] = 1
                # append command and label
                commands.append(command)
                labels.append(label)
                i += 1

    # append stop command
    commands.append('stop')
    labels.append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
    # append gripper commands
    commands.append('open the gripper')
    labels.append(np.array([0, 0, 0, 0, 1], dtype=np.float32))
    commands.append('close the gripper')
    labels.append(np.array([0, 0, 0, 0, 0], dtype=np.float32))

    # generate all combinations of command and previous command
    commands2 = commands.copy()
    commands2.append('keep moving')
    commands2.append('move back')

    # get all text embeddings from CLIP
    tokens = clip.tokenize(commands2).to(device)
    text_features = clip_model.encode_text(tokens).float()
    combo_commands = []
    samples = []
    labels_p = []
    labels2 = []
    for i in range(len(commands)): # previous
        for j in range(len(commands2)): # current
            combo_commands.append(commands2[j])
            if commands2[j] == 'keep moving':
                samples.append((labels[i], text_features[j], labels[i]))
                labels_p.append(labels[i])
                labels2.append(labels[i])
            elif commands2[j] == 'move back':
                label = -labels[i]
                label[3:] *= -1
                samples.append((labels[i], text_features[j], label))
                labels_p.append(labels[i])
                labels2.append(label)
            else:
                label2 = labels[j].copy()
                label2[-1] = labels[i][-1] # gripper should pass through
                # override label2 if gripper command
                if 'gripper' in commands2[j]:
                    label2 = labels[j]
                samples.append((labels[i], text_features[j], label2))
                labels_p.append(labels[i])
                labels2.append(label2)

    # create additional keep moving samples
    for i in range(256):
        # random x
        x = np.random.uniform(-dist, dist) * 2
        # random y
        y = np.random.uniform(-dist, dist) * 2
        # random z
        z = np.random.uniform(-dist, dist) * 2
        # random v
        v = np.random.uniform(0, dist) * 2
        # random g
        g = np.random.randint(0, 2)
        # create label
        label = np.array([x, y, z, v, g], dtype=np.float32)
        # create sample
        samples.append((label, text_features[-2], label))
        labels_p.append(label)
        labels2.append(label)
        combo_commands.append('keep moving')

    # create additional move back samples
    for i in range(256):
        # random x
        x = np.random.uniform(-dist, dist) * 2
        # random y
        y = np.random.uniform(-dist, dist) * 2
        # random z
        z = np.random.uniform(-dist, dist) * 2
        # random v
        v = np.random.uniform(0, dist) * 2
        # random g
        g = np.random.randint(0, 2)
        # create label
        label = np.array([x, y, z, v, g], dtype=np.float32)
        # create sample
        label2 = -label.copy()
        label2[3:] *= -1
        samples.append((label, text_features[-1], label2))
        labels_p.append(label)
        labels2.append(label2)
        combo_commands.append('move back')
                    
    # save to csv file
    with open('commands.csv', 'w') as f:
        # write header
        f.write('xp, yp, zp, vp, gp, command, x, y, z, v, g\n')
        for i in range(len(combo_commands)):
            # write x y and z values with 1 decimal place
            f.write(','.join(['{:.1f}'.format(x) for x in labels_p[i].tolist()]) + 
                    ',' + combo_commands[i] + ',' + 
                    ','.join(['{:.1f}'.format(x) for x in labels2[i].tolist()]) + '\n')

    return samples

# generate training commands and labels
values = ['', 'a little ', 'a lot ', 'a tiny bit ']
directions = ['left', 'right', 'forward', 'backward', 'up', 'down', 'left and forward', 
              'left and backward', 'left and up', 'left and down', 'right and forward',
              'right and backward', 'right and up', 'right and down', 'forward and up',
              'forward and down', 'backward and up', 'backward and down']
speeds = ['', ' slowly', ' quickly']
# initialize CLIP model
device = 'cuda:1'
# create the clip model
model, _ = load_clip('RN50', jit=False, device=device)
clip_model = build_model(model.state_dict())
clip_model.to(device)
del model
samples = get_dataset(values, directions, speeds, clip_model)
# create dataset object
train_dataset = LangDataset(samples)
# create data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

# initialize MLP model
l2a = L2A(h1 = 1029).to(device)

# training loop
epochs = 30
lr = 5e-4
optimizer = torch.optim.Adam(l2a.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()
train_losses = []
l2a.train()
best_loss = 1e9
for epoch in range(epochs):
    # cut learning rate after n epochs
    if epoch == 25:
        for g in optimizer.param_groups:
            g['lr'] = 1e-4
    if epoch == 40:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    # loop through data loader
    for i, (previous, command, label) in enumerate(train_loader):
        # move to device
        previous = previous.to(device)
        command = command.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # get predictions
        y, g = l2a.forward(previous, command)
        # calculate loss
        loss = l2a.loss(y, g, label)
        # backprop
        loss.backward(retain_graph = True)
        # loss.backward()
        optimizer.step()
    # print train loss
    print(f'Epoch: {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}')
    # save train loss
    train_losses.append(loss.item())
    # save the model if best loss
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(l2a.state_dict(), 'l2a.pt')

# plot losses
plt.plot(train_losses, label='train')
plt.legend()
plt.title('Losses')
ax = plt.gca()
ax.set_ylim([0, 1])
plt.show()


