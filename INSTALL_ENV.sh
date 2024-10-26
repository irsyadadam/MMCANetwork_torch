#python 3.11

conda install -y pip 

#stl
conda install -y tqdm
conda install -y pandas
conda install -y numpy
conda install -y ipykernel
conda install -y jupyter
conda install -y seaborn
conda install -y networkx
conda install -y scikit-learn
conda install -y scipy


#add to jupyter_kernel
python -m ipykernel install --user --name=MMCA_simul

#deep learning (torch 2.2, cuda 12.2)
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia

#misc
pip install missingno
pip install -U aeon
