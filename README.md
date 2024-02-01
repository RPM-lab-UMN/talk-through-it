# Talk Through It

[**Talk Through It: End User Directed Robot Learning**](https://arxiv.org)<br>
Carl Winge, Adam Imdieke, Bahaa Aldeeb, Dongyeop Kang, Karthik Desingh

Talk through it is a framework for learning robot manipulation from natural language instructions.
Given a factory model that can perform primitive actions, users can instruct the robot to perform 
more complex skills and tasks. We fine tune the factory model on saved recordings of the skills and tasks 
to create home models that can perform primitive actions as well as higher level skills and tasks.

Project website: [talk-through-it.github.io](https://talk-through-it.github.io)

## Guide (TODO)
This repository started as a clone of [PerAct](https://github.com/peract/peract). The requirements should be the same.

### 1. Environment
```bash
mamba create -n talk
mamba activate talk
mamba install python=3.8
```

## Citation