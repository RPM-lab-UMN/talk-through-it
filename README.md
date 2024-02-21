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

### 2. PyRep and Coppelia Simulator

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd <install_dir>
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).


## Citation