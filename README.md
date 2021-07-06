# event_interoperability
Repository for experiments on interoperability of semantically annotated corpora for events in English.

The pre-trained models have been obtained by using a state-of-the-art Bi-LSTM-CRF system [Reimers and Gurevych, 2017](http://aclweb.org/anthology/D17-1035). The original repository is available [here](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf), a forked version (updated at Oct. 3rd 2018) is available [here](https://github.com/tommasoc80/emnlp2017-bilstm-cnn-crf). To facilitate the reproducibility of the experiments and run the trained models on new data (from CoNLL format - use the script RunModel_CoNLL_Format.py in this repository), you should use the forked version of the repository: first, you clone the forked (or the original) repository, then you clone this repository to run the trained models. Trained models and word embeddings are available [here](https://drive.google.com/drive/folders/1HluGfwjQ4fyoVLIYEVFL24nknnHfuNg3?usp=sharing)

Training scripts for each corpus are available in the folder /training_scripts

TempEval-3 (TE3) and Meantime (MNT) data are available in the folder /data. The RED corpus is licenced via [LDC](https://catalog.ldc.upenn.edu/LDC2016T23) and cannot be re-distributed. The MNT data are licenced with a CC BY 4.0 licence.




