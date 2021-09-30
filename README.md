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

<a name="gettingstarted"></a>
## Getting Started

<a name="embeddings"></a>
### Generating Embeddings

<a name="ppi"></a>
### Running the PPI Classifier

<a name="issues"></a>
## Known Issues

<a name="conclusion"></a>
## Future Work and Conclusion
In this thesis, we developed a physical, binary Protein-Protein Interaction prediction
method. We used an experimentally proven physical protein interaction dataset derived
from APID and then utilized embeddings of each interacting protein to train a Siamese
and a Triplet Network. Various network and hyperparameter optimizations were performed to improve performance alongside the use of several different loss functions. We also sampled the dataset based on two different criterion: the "hubbiness" of certain interacting proteins, and the pairwise distance between proteins. The test set was further classified as C1, C2, C3 test sets based on the presence of an interaction partner that has a sequence-similar protein in the train set.

In the future, the integration of
evolutionary profiles gathered from methods such as Multiple Sequence Alignment (MSA) and information derived from methods like Position-Specific Scoring Matrix (PSSM), along with localization information paired with the learned embeddings can further reduce false positives.

<a name="references"></a>
## References

[1] D. Alonso-López, F. J. Campos-Laborie, M. A. Gutiérrez, L. Lambourne, M. A. Calderwood, M. Vidal, and J. De Las Rivas. “APID database: redefining protein–protein interaction experimental evidences and binary interactomes.” en. In: Database 2019 (Jan. 2019). doi: 10.1093/database/baz005.

[2] S. Chopra, R. Hadsell, and Y. LeCun. “Learning a Similarity Metric Discriminatively, with Application to Face Verification.” In: 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05). IEEE. doi: 10.1109/cvpr.2005.202.

[3] T. Hamp and B. Rost. “Evolutionary profiles improve protein–protein interaction
prediction from sequence.” In: Bioinformatics 31.12 (Feb. 2015), pp. 1945–1950. doi: 10.1093/bioinformatics/btv077.

[4] M. Heinzinger, A. Elnaggar, Y. Wang, C. Dallago, D. Nechaev, F. Matthes, and B.
Rost. “Modeling aspects of the language of life through transfer-learning protein
sequences.” In: BMC Bioinformatics 20.1 (Dec. 2019). doi: 10.1186/s12859-019-
3220-8.

[5] P. H. Le-Khac, G. Healy, and A. F. Smeaton. “Contrastive Representation Learning: A Framework and Review.” In: IEEE Access 8 (2020). arXiv: 2010.05113, pp. 193907–193934. issn: 2169-3536. doi: 10.1109/ACCESS.2020.3031549.

[6] Y. Lecun and F. Huang. “Loss Functions for Discriminative Training of EnergyBased Models.” In: Jan. 2005.

[7] Li Fei-Fei, R. Fergus, and P. Perona. “One-shot learning of object categories.”
In: IEEE Transactions on Pattern Analysis and Machine Intelligence 28.4 (Apr. 2006),
pp. 594–611. issn: 1939-3539. doi: 10.1109/TPAMI.2006.79.

[8] Y. Park and E. M. Marcotte. “Revisiting the negative example sampling problem
for predicting protein–protein interactions.” en. In: Bioinformatics 27.21 (Nov. 2011), pp. 3024–3028. issn: 1367-4803. doi: 10.1093/bioinformatics/btr514.

[9] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer. “Deep contextualized word representations.” In: (Feb. 15, 2018).
arXiv: 1802.05365 [cs.CL].

[10] C.-Y. Wu, R. Manmatha, A. J. Smola, and P. Krähenbühl. “Sampling Matters in
Deep Embedding Learning.” In: (June 23, 2017). arXiv: 1706.07567 [cs.CV].
