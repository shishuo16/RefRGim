# RefRGim
an intelligent genotype imputation reference reconstruction method with convolutional neural networks based on genetic similarity of individuals from input data and current references. RefRGim has been pretrained with single nucleotide polymorphism data of individuals in 1000 Genomes Project, which are from 26 different populations across the world. A population was delimited as a haplotype group.
## Background
Genotype imputation is a statistical method for estimating missing genotypes that are not directly assayed or sequenced in study from a denser haplotype reference panel.Existing methods usually performed well on imputing common frequency variants, but not ideal for rare variants, which typically play important roles in many complex human diseases and phenotype studies. Previous studies showed the population similarity between study and reference panel is one of the key features influencing the imputation performance. 
## Installation
### Requirements
* [Python 3.6](https://www.python.org/downloads/)
* [Tensorflow (1.9.0)](https://www.tensorflow.org/?hl=zh-cn)
* [Numpy (1.18.5)](https://numpy.org/)
* [Haplotype reference panel of the 1000 Genomes Project](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502)

### Install
**Note:** we have pre-trained our model with all variants data in the 1000 Genomes project and generated 251 convolutional neural networks (CNNs) from 22 autosomes. Considering from the aspect of saving download time and computer memory, you can only choose one chr file in 1KGP_CNN_net to download. Or if you do not care about a little more downloading time and computer memory, you can download RefRGim using default method:
```
### Download
git clone --recursive https://github.com/shishuo16/RefRGim.git
```
### Download and process reference panels of the 1000 Genomes Project
```
mkdir raw_1KGP
cd raw_1KGP
### Download
sh ../scripts/downloadfile.sh
### Process
sh ../scripts/RefRGim_process.ref.sh
```
## Running 
RefRGim takes compressed study vcf file ([VCFv4.2 format](https://samtools.github.io/hts-specs/VCFv4.2.pdf)), RefRGim path, path of raw reference panels of the 1000 Genomes Project, and a prefix for output files as inputs and output the study specified reference panels, most genetic-similar haplotype group for each input individuals, the genetic-similar probability matrix of haplotype groups and input individuals, retrained convolutional neural network, and convolutional neural network retraining process.
### Demo
```
./RefRGim example/test.vcf.gz ./ raw_1KGP example/test.out
```
### Outputs 
- test.out.SuperPopulation/chr*.vcf.gz
    - study specified reference panels. Haplotypes whose population belongs to a same super population were merged into one vcf file
- test.out.populations
    - study individual classification result: individual name, haplotype group, and super haplotype group
- test.out.population.probs
    - probability matrix of input individuals and 26 haplotype group
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

