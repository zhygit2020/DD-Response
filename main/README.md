# DD-Response Usage
##### Jiayi Yin <sup>†</sup>, Hanyu Zhang <sup>†</sup>, Xiuna Sun, Nanxin You, Minjie Mou, Fengcheng Li, Honglin Li<sup> * </sup>, Su Zeng<sup> * </sup>, and Feng Zhu<sup> * </sup>



## Dependencies

- DD-Response should be deployed on Linux in python 3.6.
- Main requirments: `python==3.6.8`, `pytorch==1.8.1`, `captum==0.5.0`, `lapjv==1.3.1`, `umap-learn==0.3.10`, `RDkit==2020.09.5`, `scikit-learn==0.23.0`, `scipy==1.1.0`.
- To use GPU, please install the GPU version of  `pytorch`.



## Usage

### (A). Predict new lncRNA-miRNA interactions using pre-trained models with pre-built graph and original LMI data
#### 1. For any miRNA or lncRNA users want to investigate
##### 1.1 Place the name of the RNA that users want to investigate into the `./data/run_data/target_rna.csv` imitating the examples. 
​		Practicable RNAs could refer to the files of `./data/processed_data/pair_unique_RNAs/mirna/mirna_names_unique_template.csv` and `./data/processed_data/pair_unique_RNAs/lncrna/lncrna_names_unique_template.csv`. By default, `./data/run_data/target_rna.csv` contains 20 miRNAs as templates.

##### 1.2 Apply the original graph and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 256 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type run --run_mode predict_on_rna --gpu -1 
```
​		following parameters stipulate the specific graph and the pre-trained model to apply:
- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 256.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is "RNA_intrinsic".

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

  following parameters stipulate the mode while the program running:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "run"

- `--run_mode`, the program would run under the mode of "test", "predict_on_rna" or "predict_on_pair". Here the parameter should be set to "predict_on_rna".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ the output  will be under the automatically generated `./run_result/prerna/` directory, each .csv file contains ranked logits of candidates against the target RNAs.
#### 2. For any specific lncRNA-miRNA pairs users want to investigate
##### 2.1 Place the names of specific lncRNA-miRNA pairs that users want to investigate into the `./data/run_data/target_pair.csv` imitating the examples. 
​		Practicable RNAs could refer to the default files introduced above. If users have priori information of those pairs, labels should be placed under column "Label". Otherwise, fill the blanks with "-1".
##### 2.2 Apply the original graph and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 256 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type run --run_mode predict_on_pair --gpu -1 
```
​		following parameters stipulate the specific graph and the pre-trained model to apply:
- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 256.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is "RNA_intrinsic".

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

  following parameters stipulate the mode while the program running:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "run".

- `--run_mode`, the program would run under the mode of "test", "predict_on_rna" or "predict_on_pair". Here the parameter should be set to "predict_on_pair".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ the output  will be under the automatically generated `./run_result/prepair/` directory, the .csv file of `logits_of_prepair.csv` contains ranked logits of the evaluating pairs. For example, the results of 3 LM-pairs investigated in this study are provided. 
  
  

### (B). Retrain a brand new ncRNAInter using in-house data

#### 1. Prepare your in-house data and generate RNA features
##### 1.1 Prepare the in-house data, including the sequences of the RNAs and their interaction labels, that users want to investigate into the `./data/original_data/rna_pairs_user.csv` imitating the examples. 
##### 1.2 Go to the directory of `./data_process/`, carry out the data pre-processing and the comprehensive RNA feature generation by executing the following command: 
```
python data_process.py --random_seed 42 --data_define user --negative_sampling True --test_rate 0.1 
```
​		following parameters stipulate the specific graph and the pre-trained model to apply:
- `--random_seed`, set random seed, default is 42.

- `--data_define`, set the data source, default is "self", which means the processing data is defined by this research. When users want to process their own data in `rna_pairs_user.csv`, this parameter should be set as "user".

- `--negative_sampling`, run negative sampling if needed, default is True.

- `--test_rate`, the rate for test dataset splitting, default is 0.1, it will not split out test data when set as 0.

  __Output:__ the output  will be under the automatically generated `./data/processed_data/` directory, including the RNA features as well as the interacting pairs for training and testing. 

#### 2. Apply source codes to train your own ncRNAInter
##### 2.1 Carry out the graph building, model constructing and cross-validating by executing the following command:
```
python ./K_fold_trainval.py --lr 5e-4 --hidden_dim 256 --n_layers 2 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type trainval --gpu -1 
```
​		following parameters stipulate the specific graph and the pre-trained model to apply:
- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 256.

- `--n_layers`, set the number of hidden NodeSAGE layers, default is 2.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is "RNA_intrinsic".

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

  following parameters stipulate the mode while the program running:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "trainval".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ the output will be under the automatically generated `./pretrained/` directory. 



## Citation and Disclaimer

The manuscript is currently under peer review. Should you have any questions, please contact Dr. Zhang at hanyu_zhang@zju.edu.cn

