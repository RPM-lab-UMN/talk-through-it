#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install PyRep
echo -e '\nCloning PyRep...\n'
cd ${SCRIPT_DIR}
git clone https://github.com/stepjam/PyRep.git

cd PyRep
echo -e '\nInstalling PyRep requirements...\n'
pip install -r requirements.txt -q
pip install .

# Installing peract require,ents
echo -e '\nInstalling peract requirements...\n'
cd ${SCRIPT_DIR}
pip install -r requirements.txt -q
pip install git+https://github.com/openai/CLIP.git -q

# Install RLBench
echo -e '\nClonoing RLBench...\n'
cd ${SCRIPT_DIR}
git clone -b peract git@github.com:RPM-lab-UMN/RLBench.git # note: 'peract' branch

cd RLBench
echo -e '\nInstalling RLBench requirements...\n'
pip install -r requirements.txt -q
python setup.py develop

# Install YARR
echo -e '\nClonoing YARR...\n'
cd ${SCRIPT_DIR}
git clone -b peract git@github.com:RPM-lab-UMN/YARR.git # note: 'peract' branch

echo -e '\nInstalling YARR requirements...\n'
cd YARR
pip install PyYAML==5.1 -q
pip install -r requirements.txt -q
python setup.py develop

# Install peract
cd ${SCRIPT_DIR}
export PERACT_ROOT=$(pwd)  # mostly used as a reference point for tutorials
python setup.py develop

# Return to root directory
[ ! -z "$PROJECT_ROOT" ] && cd $PROJECT_ROOT