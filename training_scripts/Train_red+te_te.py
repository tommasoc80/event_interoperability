# This script trains the BiLSTM-CRF architecture for part-of-speech tagging
# and stores it to disk. Then, it loads the model to continue the training.
# For more details, see docs/Save_Load_Models.md
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle



# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'red_te100_cross_te':                            #Name of the dataset
        {'columns': {0:'tokens', 1:'POS', 2:'dep', 3:'Event_BIO'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'Event_BIO',                     #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}

# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
embeddingsPath = '/home/p281734/projects/event_extraction_cross-dataset_workspace/emnlp2017-bilstm-cnn-crf-py2/Komninos_Wikipedia_WordEmbeddings_only_words'#'komninos_english_embeddings.gz'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
#params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25)}
params = {'dropout': [0.5, 0.5], 'classifier': 'CRF', 'LSTM-Size': [100], 'optimizer': 'nadam', 'charEmbeddings': 'CNN', 'miniBatchSize': 8}
#params = {'dropout': [0.5, 0.5], 'classifier': 'CRF', 'LSTM-Size': [100, 100], 'optimizer': 'nadam', 'charEmbeddings': 'CNN', 'miniBatchSize': 8}

#print("Train the model with 1 Epoch and store to disk")
model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.modelSavePath = "models/red+te_te_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=30)
