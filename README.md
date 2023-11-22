# <u>D</u>ecoding <u>D</u>rug <u>Response</u> (DD-Response) with Structurized Gridding Map (SGM)-based Cell Representation

##### Jiayi Yin <sup>†</sup>, Hanyu Zhang <sup>†</sup>, Xiuna Sun, Nanxin You, Minjie Mou, Fengcheng Li , Ziqi Pan, Honglin Li<sup> * </sup>, Su Zeng<sup> * </sup>, and Feng Zhu<sup> * </sup>



## Graphic Abstract

 [graphic_abstract](./paper/materials/graphic_abstract.png) 



## Model Architecture

 [model_architecture](./paper/materials/model_architecture.png) 



## Dependencies

- DD-Response should be deployed on Linux in python 3.6.
- Main requirments: `python==3.6.8`, `pytorch==1.8.1`, `captum==0.5.0`, `lapjv==1.3.1`, `umap-learn==0.3.10`, `RDkit==2020.09.5`, `scikit-learn==0.23.0`, `scipy==1.1.0`.
- To use GPU, please install the GPU version of  `pytorch`.



## Install

1. Download source codes of DD-Response.
2. DD-Response should be deployed on Linux.
3. The DD-Response tree includes three directories:

```
 |- main
     |- bashes
     |- data
     |- feamap
     |- model
     |- run
     |- 0_feadist_fit.py
     |- 0_map_transfer.py
     |- 0_split_cvdata.py
     |- main.py
     |- tcga_main.py
 |- paper
    |- interpretation
 |- README.md
 |- LICENSE
```

The directory of `main` deposits the basis of DD-Response, including source code and datasets. 

The directory of `paper` deposits the necessary scripts for analyses in the paper.



## Usage

#### 1. To train a model

##### 1.1 Place the training data that users want to investigate into the `.main/data/original_data/` imitating the examples. 

##### 1.2 Execute the following bash commands in the directory of `.main/bashes`:

```
sh 0_split_cvdata.sh		# data splitting for cross-validation
```
```
sh 0_trans_cell.sh		# RGM representation transform for cell lines
sh 0_trans_drug.sh		# RGM representation transform for drugs
```
```
sh DRS_molossbt128.sh	# model Training through cross-validation
```
__Output:__ the output will be under the automatically generated `./main/data/processed_data` directory and `./main/pretrain_data/` directory.

##### 1.3 If users want to reconstruct their own RGM template, Execute the following bash commands in the directory of `.main/bashes` before RGM representation transform: 

```
sh 0_feadist.sh		# calculate the scales as config files for RGM template construction
```

__Output:__ the output will be under the automatically generated `./main/data/processed_data` directory.

__Note:__ the output `.cfg` files should be manually moved to `./main/feamap/config/trans_from_ALL` before running `0_trans_cell.sh` and `0_trans_drug.sh`

#### 2. To predict samples using the pre-trained model

##### 2.1 Place the predicting data that users want to investigate into the `./data/predict_data/` imitating the examples.

##### 2.2 Execute the following bash commands in the directory of `.main/bashes`:

```
sh Predict_gCSI.sh	# Run the model for gCSI data prediction
```

__Output:__ the output will be under the automatically generated `./main/data/predict_result` directory.

#### 3. To construct the model based on TCGA dataset

##### 3.1 Place the training data that users want to investigate into the `.main/transfer/data/original_data/` imitating the examples. 

##### 3.2 Execute the following bash commands in the directory of `.main/bashes`:

```
sh TCGA_modeling.sh		# model Training through cross-validation
```

__Output:__ the output will be under the automatically generated `./main/transfer/data/processed_data` directory and `./main/transfer/pretrained` directory.



## Citation and Disclaimer

The manuscript is currently under peer review. Should you have any questions, please contact Dr. Zhang at hanyu_zhang@zju.edu.cn

