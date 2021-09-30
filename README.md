<h1><b>Protein-Protein Interaction Prediction using Contrastive Learning</b></h1>

## Abstract

Contrastive Learning methods have recently received interest due to their performance
in representation learning tasks in the Computer Vision domain. However, there
has not been much application of them in the realm of Bioinformatics, especially
Protein Bioinformatics. In this work, we apply contrastive learning approaches to predict direct/physical interaction between proteins based on sequences alone (without the use of any evolutionary data). The proteins are represented as vectors derived from the new Language Models trained on protein sequences. Then, these representations of the interacting proteins are utilized in a Contrastive Learning paradigm along with various distance metrics and specific Loss functions to differentiate between (physically) interacting and non-interacting protein pairs.
# Table of Contents
* [Architecture Overview](#Architecture_Overview)
* [Datasets](#datasets)
* [Theoretical Foundations](#theory)
    * [SeqVec Embeddings](#seqvec)
    * [Contrastive Learning](#contrastive)
    * [Siamese Networks](#siamese)
    * [Hard-Negative Mining](#negative)
* [Getting Started](#gettingstarted)
    * [Generating Embeddings](#embeddings)
    * [Running the PPI classifier](#ppi)
* [Known Issues](#issues)
* [Future Work and Conclusion](#conclusion)
* [References](#references)

<a name="Architecture_Overview"></a>
## Architecture Overview

<a name="datasets"></a>
## Datasets

<a name="theory"></a>
## Theoretical Background

<a name="seqvec"></a>
### SeqVec Embeddings

<a name="contrastive"></a>
### Contrastive Learning

<a name="siamese"></a>
### Siamese Networks

<a name="negative"></a>
### Hard-Negative Mining