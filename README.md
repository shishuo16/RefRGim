# RefRGim
an intelligent genotype imputation reference recommendation method with convolutional neural networks based on genetic similarity of individuals from input data and current references. RefRGim has been pretrained with single nucleotide polymorphism data of individuals in 1000 Genomes Project, which are from 26 different populations across the world.
## Background
Genotype imputation is a statistical method for estimating missing genotypes that are not directly assayed or sequenced in study from a denser haplotype reference panel.Existing methods usually performed well on imputing common frequency variants, but not ideal for rare variants, which typically play important roles in many complex human diseases and phenotype studies. Previous studies showed the population similarity between study and reference panel is one of the key features influencing the imputation performance. 
## Installation
### Requirements
* [Python 3.6](https://www.python.org/downloads/)
* [Tensorflow (1.9.0)](https://www.tensorflow.org/?hl=zh-cn)
* [Numpy (1.18.5)](https://numpy.org/)
### Install
**Note:** we have pre-trained our model with all variants data in the 1000 Genomes project and generated 252 convolutional neural networks (CNNs) from 22 chromosome. In practice, You only need download CNNs from one chromosome. Considering from the aspect of saving download time and memory, you can only choose one chr file in 1KGP_CNN_net to download. Or if you do not care about a little more downloading time and computer memory, you can download RefRGim using: 
Default method to install:
```
### Download
git clone --recursive https://github.com/shishuo16/RefRGim.git
```
## Running 
RefRGim takes standard vcf file ([VCFv4.2 format](https://samtools.github.io/hts-specs/VCFv4.2.pdf)) and a prefix as inputs and output the population identification results of input individuals, the probabilities of each population class of individuals belonged to, retrained convolutional neural network, and convolutional neural network retraining info.
### Demo
```
python3 RefRGim.py example/test.vcf example/test.out ${RefRGim_path}
```
### Outputs 
- test.out.populations
    - individual classification result: individual name, population class, and super population class
- test.out.population.probs
    - probability matrix of individuals and 26 populations
- test.out_net
    - directory that saves the retrained weights and parameters for the model
- test.out_training_info
    - directory that saves graph of weights, biases, and loss function in retraining process, which can be display using [tensorboard](https://www.tensorflow.org/tensorboard?hl=zh-cn):
    ```
    tensorboard --logdir=test.out_training_info
    ```
## Maintainers
[@shishuo16](https://github.com/shishuo16)
## Citations
To be continued ...

