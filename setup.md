# Environment Setup
- Create conda environment named conv (you can replace the name)
```bash
conda create -n theta python=3.8.12
```

- Activate environment
```bash
conda activate theta
```

- Install CUDA 11.6
```bash
conda install -c conda-forge cudatoolkit=11.6.0
```

- Solve Conda Python path issue
https://stackoverflow.com/questions/62352699/conda-uses-local-packages

```bash
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d/
cat > ${CONDA_PREFIX}/etc/conda/activate.d/disable-PYTHONUSERBASE.sh << EOF
#!/bin/bash
export PYTHONUSERBASE=intentionally-disabled
EOF

chmod +x ${CONDA_PREFIX}/etc/conda/activate.d/disable-PYTHONUSERBASE.sh
```

- Install PyTorch 1.13

```bash
pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

- Install Apex

```bash
# Option 1
# Change the path CUDA_HOME if needed.
# export CUDA_HOME=/sw/cuda/11.6.0
git clone https://github.com/boru-roylu/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

You can also download apex directly from NVIDIA repository. After cloning the repository, checkout out to `b88c507` and comment line 175 in setup.py to ignore checking cuda version mismatch.
```bash
# Option 2
git clone https://github.com/NVIDIA/apex.git
cd apex
git co b88c507
```

- Install Python packages
```bash
pip install --no-cache-dir -r requirements.txt
```

- Install Jupyter Notebook
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=theta
```