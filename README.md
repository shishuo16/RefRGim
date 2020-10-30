# HGImp
an intelligent genotype imputation reference recommendation method with convolutional neural networks based on genetic similarity of individuals from input data and current references. HGImp has been pretrained with single nucleotide polymorphism data of individuals in 1000 Genomes Project, which are from 26 different populations across the world .
## Background
Genotype imputation is a statistical method for estimating missing genotypes that are not directly assayed or sequenced in study from a denser haplotype reference panel.Existing methods usually performed well on imputing common frequency variants, but not ideal for rare variants, which typically play important roles in many complex human diseases and phenotype studies. Previous studies showed the population similarity between study and reference panel is one of the key features influencing the imputation performance. 
## Installation
### Requirements
* [python3.6](https://www.python.org/downloads/)
* [Tensorflow (1.9.0+)](https://www.tensorflow.org/?hl=zh-cn)
* [Numpy (1.18.5+)](https://numpy.org/)
### Install
Default method to install:
```
### Download
git clone --recursive https://github.com/shishuo16/HGImp.git
### Configure
vim ~/.bashrc
export HGImp_PATH="(Absolute_path_HGImp_located_in)/HGImp/"
source ~/.bashrc
```
## Running  
HGImp takes standard vcf file ([VCFv4.2 format](https://samtools.github.io/hts-specs/VCFv4.2.pdf)) and a prefix as inputs and output the population identification results of input individuals and the probabilities of each population class of individuals belonged to:
'''
python3 $HGImp_PATH/HGImp.py example/test.vcf example/test.out
'''
### Outputs 
- test.out.populations
 - individual classification result: individual name, population class, and super population class
- test.out.population.probs
 - probability matrix of individuals and 26 populations
  
