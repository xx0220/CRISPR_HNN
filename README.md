# CRISPR_HNN
Prediction of CRISPR-Cas9 On-Target Activity Based on a Hybrid Neural Network 

Below is the layout of the whole model.
# Environment
* Python 3.7.12
* TensorFlow-GPU 2.5.0
* cudatoolkit 11.3.1
* numpy 1.18.5
* pandas 1.3.5
# Datasets
Include 9 public datasets:
* WT
* ESP
* HF
* xCas
* SpCas9
* Sniper
* HCT116
* HELA
* HL60
# File description
* model.py: Crispr_HNN model overall architecture
* model_train.py: Running this file to train the Crispr_HNN model. (5-fold cross-validation)
* model_test.py: Running this file to evaluate the Crispr_HNN model. (Demonstrate model performance by evaluating metrics through two regression question evaluation indicators)
* csvtopkl.py: Process the dataset by calling csvtopkl.py to obtain the encoding format required by CRISPR_HNN.
* api.py: A Flask-based API service for CRISPR_HNN prediction. Receives input sequences and selected database, loads the corresponding model, and returns prediction results.
* index.html: A simple web interface to interact with the CRISPR_HNN API. Users can input sequences, select datasets, and view predictions directly in the browser.
