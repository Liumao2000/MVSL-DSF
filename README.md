# The associated data and preliminary code have been uploaded to the repository. Upon successful submission and completion of the peer review process, the full and finalized version of the core code will be made publicly available alongside the published article.    



# MVSL-DSF
Multiview Subspace Representation Learning and Cross-modal Feature Dynamic Aggregation for Enhanced Drug Side Effect Frequency Prediction
# Requirements  
dgl-cu113                 0.7.2                   
dgllife                   0.2.6    
scikit-learn              1.0.2  
rdkit                     2023.3.2   
python                    3.7.1  
torch                     1.10.0+cu113   
# data 
drug_side_association_matrix.pckl: the known drug-ADR interaction matrix. 

drug_side_frequency_matrix.pkl: the frequency drug-ADR interaction matrix. 
 
drug_smiles.pkl: the smiles sequences of drugs. 

final_sample.pkl: all known drug-ADR interactions and corresponding labels. 

side_vector_level_123.pkl: Semantic feature vectors of ADRs. 


# Code  
Network.py: This function contains the network framework of our entire model and is based on pytorch 1.10.   

Cross_validation.py: This function can test the predictive performance of our model under ten-fold cross-validation.  

# Train and test folds
python cross_validation.py --rawdata_dir /Your path --num_epochs Your number --batch_size Your number 

rawdata_dir: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.) 

num_epochs: Define the maximum number of epochs. 

batch_size: Define the number of batch size for training and testing. 

All files of Data and Code should be stored in the same folder to run the model. 

Example:

python cross_validation.py --rawdata_dir /data --num_epochs 50 --batch_size 128 

# Contact
If you have any questions or suggestions with the code, please let us know. Contact Liumao at Liumao2000@foxmail.com   
