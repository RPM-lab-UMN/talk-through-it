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
    device = 'cuda:0'
    i = 0
    commands = []
    labels = []
    dist = 10
    label_len = 8
    for i in range(2):
        for d in directions:
            for v in values:
                for s in speeds:
                    # generate command
                    command = 'move ' + v + d + s
                    # generate the label
                    label = np.zeros(label_len, dtype=np.float32)
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
                    # if 'and' in d:
                    #     label = label / 2**0.5
                    # set the speed label
                    label[3] = dist
                    if 'slowly' in s:
                        label[3] = dist / 2
                    elif 'quickly' in s:
                        label[3] = dist * 2
                    # alternate gripper label
                    # TODO double instead of alternating
                    if i == 0:
                        label[-1] = 1
                    # append command and label
                    commands.append(command)
                    labels.append(label)

    # append command for 'move a lot backward and a little up'
    for i in range(2):
        for d1 in directions:
            for d2 in directions:
                for v1 in values:
                    for v2 in values:
                        # generate command
                        command = 'move ' + v1 + d1 + ' and ' + v2 + d2
                        # generate the label
                        label = np.zeros(label_len, dtype=np.float32)
                        # first part
                        if 'left' in d1:
                            i1 = 0
                            i2 = 1
                            i3 = 2
                            if 'left' in d2:
                                continue
                            elif 'right' in d2:
                                continue
                            label[i1] = -dist
                        elif 'right' in d1:
                            i1 = 0
                            i2 = 1
                            i3 = 2
                            if 'left' in d2:
                                continue
                            elif 'right' in d2:
                                continue
                            label[i1] = dist
                        if 'forward' in d1:
                            i1 = 1
                            i2 = 0
                            i3 = 2
                            if 'forward' in d2:
                                continue
                            elif 'backward' in d2:
                                continue
                            label[i1] = dist
                        elif 'backward' in d1:
                            i1 = 1
                            i2 = 0
                            i3 = 2
                            if 'forward' in d2:
                                continue
                            elif 'backward' in d2:
                                continue
                            label[i1] = -dist
                        if 'up' in d1:
                            i1 = 2
                            i2 = 0
                            i3 = 1
                            if 'up' in d2:
                                continue
                            elif 'down' in d2:
                                continue
                            label[i1] = dist
                        elif 'down' in d1:
                            i1 = 2
                            i2 = 0
                            i3 = 1
                            if 'up' in d2:
                                continue
                            elif 'down' in d2:
                                continue
                            label[i1] = -dist
                        if 'a little' in v1:
                            label = label / 2
                        elif 'a lot' in v1:
                            label = label * 2
                        elif 'a tiny bit' in v1:
                            label = label / 10

                        # second part
                        if 'left' in d2:
                            label[0] = -dist
                        elif 'right' in d2:
                            label[0] = dist
                        if 'forward' in d2:
                            label[1] = dist
                        elif 'backward' in d2:
                            label[1] = -dist
                        if 'up' in d2:
                            label[2] = dist
                        elif 'down' in d2:
                            label[2] = -dist
                        # modify label based on value
                        if 'a little' in v2:
                            label[i2] = label[i2] / 2
                            label[i3] = label[i3] / 2
                        elif 'a lot' in v2:
                            label[i2] = label[i2] * 2
                            label[i3] = label[i3] * 2
                        elif 'a tiny bit' in v2:
                            label[i2] = label[i2] / 10
                            label[i3] = label[i3] / 10
                        # set the speed label
                        label[3] = dist
                        # alternate gripper label
                        if i == 0:
                            label[-1] = 1

                        # append command and label
                        commands.append(command)
                        labels.append(label)

    # append commands for rotate clockwise and counterclockwise
    directions = ['clockwise', 'counterclockwise']
    for i in range(2):
        for d in directions:
            for v in values:
                commands.append('rotate ' + v + d)
                label = np.zeros(label_len, dtype=np.float32)
                label[4] = 90
                if 'a little' in v:
                    label = label / 2
                elif 'a lot' in v:
                    label = label * 2
                elif 'a tiny bit' in v:
                    label = label / 10
                if 'counterclockwise' in d:
                    label = -label
                if i == 0:
                    label[-1] = 1
                labels.append(label)

    # append commands for turn left and right
    directions = ['left', 'right']
    for i in range(2):
        for d in directions:
            for v in values:
                commands.append('turn ' + v + d)
                label = np.zeros(label_len, dtype=np.float32)
                label[6] = 90
                if 'a little' in v:
                    label = label / 2
                elif 'a lot' in v:
                    label = label * 2
                elif 'a tiny bit' in v:
                    label = label / 10
                if 'left' in d:
                    label = -label
                if i == 0:
                    label[-1] = 1
                labels.append(label)

    # append commands for turn up and down
    directions = ['up', 'down']
    for i in range(2):
        for d in directions:
            for v in values:
                commands.append('turn ' + v + d)
                label = np.zeros(label_len, dtype=np.float32)
                label[5] = 90
                if 'a little' in v:
                    label = label / 2
                elif 'a lot' in v:
                    label = label * 2
                elif 'a tiny bit' in v:
                    label = label / 10
                if 'up' in d:
                    label = -label
                if i == 0:
                    label[-1] = 1
                labels.append(label)

    # append stop command
    commands.append('stop')
    label = np.zeros(label_len, dtype=np.float32)
    labels.append(label)
    # append gripper commands
    commands.append('open the gripper')
    label = np.zeros(label_len, dtype=np.float32)
    label[-1] = 1
    labels.append(label)
    commands.append('close the gripper')
    label = np.zeros(label_len, dtype=np.float32)
    labels.append(label)

    # save to csv file
    with open('commands.csv', 'w') as f:
        # write header
        f.write('command, x, y, z, v, r, p, yaw, g\n')
        for i in range(len(commands)):
            # write x y and z values with 1 decimal place
            f.write(commands[i] + ',' + 
                    ','.join(['{:.1f}'.format(x) for x in labels[i].tolist()]) + '\n')

    # generate all combinations of command and previous command
    commands2 = commands.copy()
    commands2.append('keep moving')
    commands2.append('move back')

    # get all text embeddings from CLIP
    with torch.no_grad():
        tokens1 = clip.tokenize(commands2[:500]).to(device).detach()
        tokens2 = clip.tokenize(commands2[500:]).to(device).detach()
        text_features1 = clip_model.encode_text(tokens1).detach().float().cpu()
        text_features2 = clip_model.encode_text(tokens2).detach().float().cpu()
        text_features = torch.cat((text_features1, text_features2), dim=0).numpy()
    combo_commands = []
    samples = []
    labels_p = []
    labels2 = []
    for i in range(len(commands)): # previous
        for j in range(len(commands2)): # current
            combo_commands.append(commands2[j])
            if commands2[j] == 'keep moving':
                samples.append((labels[i], text_features[j], labels[i], commands2[j]))
                labels_p.append(labels[i])
                labels2.append(labels[i])
            elif commands2[j] == 'move back':
                label = -labels[i]
                label[4:] *= -1
                samples.append((labels[i], text_features[j], label, commands2[j]))
                labels_p.append(labels[i])
                labels2.append(label)
            else:
                # only take 25% of this type
                skip = False
                if np.random.uniform() > 0.25:
                    skip = True
                label2 = labels[j].copy()
                label2[-1] = labels[i][-1] # gripper should pass through
                # override label2 if gripper command
                if 'gripper' in commands2[j]:
                    label2 = labels[j]
                    skip = False
                if not skip:
                    samples.append((labels[i], text_features[j], label2, commands2[j]))
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
        # random roll
        r = np.random.randint(-180, 180)
        # random pitch
        p = np.random.randint(-180, 180)
        # random yaw
        yaw = np.random.randint(-180, 180)
        # random g
        g = np.random.randint(0, 2)
        # create label
        label = np.array([x, y, z, v, r, p, yaw, g], dtype=np.float32)
        # create sample
        samples.append((label, text_features[-2], label, 'keep moving'))
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
        # random roll
        r = np.random.randint(-180, 180)
        # random pitch
        p = np.random.randint(-180, 180)
        # random yaw
        yaw = np.random.randint(-180, 180)
        # random g
        g = np.random.randint(0, 2)
        # create label
        label = np.array([x, y, z, v, r, p, yaw, g], dtype=np.float32)
        # create sample
        label2 = -label.copy()
        label2[-1] *= -1
        samples.append((label, text_features[-1], label2, 'move back'))
        labels_p.append(label)
        labels2.append(label2)
        combo_commands.append('move back')
                    
    # save to csv file
    # with open('commands.csv', 'w') as f:
    #     # write header
    #     f.write('xp, yp, zp, vp, gp, command, x, y, z, v, g\n')
    #     for i in range(len(combo_commands)):
    #         # write x y and z values with 1 decimal place
    #         f.write(','.join(['{:.1f}'.format(x) for x in labels_p[i].tolist()]) + 
    #                 ',' + combo_commands[i] + ',' + 
    #                 ','.join(['{:.1f}'.format(x) for x in labels2[i].tolist()]) + '\n')

    return samples

# generate training commands and labels
values = ['', 'a little ', 'a lot ', 'a tiny bit ']
directions = ['left', 'right', 'forward', 'backward', 'up', 'down']
speeds = ['', ' slowly', ' quickly']
# initialize CLIP model
device = 'cuda:0'
# create the clip model
model, _ = load_clip('RN50', jit=False, device=device)
clip_model = build_model(model.state_dict())
clip_model.to(device)
del model
samples = get_dataset(values, directions, speeds, clip_model)
# create dataset object
train_dataset = LangDataset(samples)
# create data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

# initialize MLP model
l2a = L2A(h1 = 1032).to(device)
# load weight from file
l2a.load_state_dict(torch.load('l2a.pt'))
l2a.eval()

# loop through data loader
success = 0
total = 0
for i, (previous, command, label, text) in enumerate(train_loader):
    # move to device
    previous = previous.to(device)
    command = command.to(device)
    label = label.to(device)
    # get predictions
    y, g = l2a.forward(previous, command)
    # print command, y, g, label if not successful
    if not l2a.success(y, g, label):
        print('command:', text)
        print('y:', y)
        print('g:', g)
        print('label:', label)
    else:
        success += 1
    total += 1

print('Success rate: {:.2f}%'.format(success / total * 100))
