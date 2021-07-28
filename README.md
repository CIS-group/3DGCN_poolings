# 3DGCN_poolings: Three-Dimensionally Embedded Graph Convolutional Network (3DGCN) for Molecule Interpretation with four different pooling operations: max, sum, avg, and set2set

This is an implementation of our paper "Effects of Pooling Operations on Prediction of Ligand Rotation-Dependent Protein-Ligand Binding in 3D Graph Convolutional Network":

Yeji Kim, Jihoo Kim, Won June Kim, Eok Kyun Lee, Insung S. Choi, [Effects of Pooling Operations on Prediction of Ligand Rotation-Dependent Protein-Ligand Binding in 3D Graph Convolutional Network] (Bull. Korean Chem. Soc. 2021, 42(5), 744-747.)


## Requirements

* Python 3.6.1
* Tensorflow 1.15
* Keras 2.25
* RDKit
* scikit-learn

## Data

* BACE dataset for Binding-activeness classification

## Models

The `models` folder contains python scripts for building, training, and evaluation of the 3DGCN model with four different pooling operations (max, sum, avg, and set2set).

The 'dataset.py' cleans and prepares the dataset for the model training.
The 'layer.py' and 'model.py' build the model structure.
The 'loss.py' and 'callbacks.py' assign the loss and metrics that we wanted to use.
The 'trainer.py', 'bace_train.py', and 'bace_eval.py' are for training and evaluation of the model.
The 'rotational_invariance.py' evaluates the trained model with ligand rotations.
