# Multi-Modal Cross Attention Network

Pytorch Implementation of Transformer Module for [Multi Modal Cross Atttention (Wei et al. 2020, *CVPR*)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf). Intention for integration into Thryroid Segmentation Fusion for Cine Classification (UCLA F24 - BE223A).

**Dataset** 

Dataset used for testing the MMCA module for time series classification is the BLINK dataset originally introduced in [(Chicaiza et al. 2021, *IEEE*)](https://ieeexplore.ieee.org/document/9590711)

- Train Size: 500
- Test Size: 450
- Sequence Length: 510
- Number of Classes: 2
- Number of Dimensions: 4

The 4 dimensions were split into 2 datasets, each of 2 elements, simulating 2 different modalities for the input channels of MMCA network. 

Data loading is done following the standard using aeon: [(notebook)](https://github.com/aeon-toolkit/aeon/blob/main/examples/datasets/data_loading.ipynb)

# 

### Install Env:

<code>
conda create -n MMCA_simul python=3.11

bash INSTALL_ENV.sh
</code>

Dependencies: 
- python = 3.11
- cuda 11.8
- torch 2.2.1
- torchvision = 0.17.1
- torchaudio = 2.2.1
- torch sparse

#

### Run: