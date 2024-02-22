# Talk Through It

[**Talk Through It: End User Directed Robot Learning**](https://arxiv.org)<br>
Carl Winge, Adam Imdieke, Bahaa Aldeeb, Dongyeop Kang, Karthik Desingh

Talk through it is a framework for learning robot manipulation from natural language instructions.
Given a factory model that can perform primitive actions, users can instruct the robot to perform 
more complex skills and tasks. We fine tune the factory model on saved recordings of the skills and tasks 
to create home models that can perform primitive actions as well as higher level skills and tasks.

Project website: [talk-through-it.github.io](https://talk-through-it.github.io)

## Guide
This repository started as a clone of [PerAct](https://github.com/peract/peract). The requirements should be the same.

Please open an issue if you encounter problems.

### 1. Environment
```bash
mamba create -n talk
mamba activate talk
mamba install python=3.8
```

### 2. PyRep and Coppelia Simulator

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

__Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

Finally install the python library:

```bash
pip3 install -r requirements.txt
pip3 install .
```

### 3. RLBench

PerAct uses our [RLBench fork](https://github.com/RPM-lab-UMN/RLBench/tree/peract). 

```bash
cd <install_dir>
git clone -b peract https://github.com/RPM-lab-UMN/RLBench.git # note: 'peract' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```

### 4. YARR

PerAct uses our [YARR fork](https://github.com/RPM-lab-UMN/YARR/tree/peract).

```bash
cd <install_dir>
git clone -b peract https://github.com/RPM-lab-UMN/YARR.git # note: 'peract' branch

cd YARR
pip install -r requirements.txt
python setup.py develop
```

### 5. Talk Through It Repo
Clone:
```bash
cd <install_dir>
git clone https://github.com/RPM-lab-UMN/talk-through-it.git
```

Install:
```bash
cd talk-through-it
pip install git+https://github.com/openai/CLIP.git
mamba install einops pytorch3d transformers
```
## Quickstart
Generate Level-1 motions data using RLBench/tools/dataset_generator.py

Train the observation-dependent model by editing conf/config.yaml and running train.py

Train the observation-independent model by running train_l2a.py and train_classifier.py

Collect demonstrations using language by running record_model_1.py

Evaluate observation-dependent models by editing conf/eval.yaml and running eval.py

## Citation
```
@misc{winge2024talk,
      title={Talk Through It: End User Directed Manipulation Learning}, 
      author={Carl Winge and Adam Imdieke and Bahaa Aldeeb and Dongyeop Kang and Karthik Desingh},
      year={2024},
      eprint={2402.12509},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```