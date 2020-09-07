## GNN_implemented

Codes using GNN to predict instable units(node classification)

### Dependencies



* Python 3.7
* Pytorch 1.5.0  cudatoolkit=10.2
* ray-psops(private)
* ray 0.8.6
* gym 0.9.1
* dgl 0.5.1
* pytorch-geometric 1.6.1

Other versions may also work, but the ones listed are the ones I've used

### Data processing

Using ray-psops to batch generate samples
Data preprocess:
* delete self-loop
* using amplitute insted of the real and imaginary parts
* label data using power angle of generation

### program runnning
#### train a model
In train.py, you can choose a model in 'Model' folder to train a weight(Baseline, only test for Baseline), 
change the hyperparameter in parser if needed, set the train set and evaluate set.
#### test a trained model
In test.py, input the weight path to test the accuracy and confusion matrix
#### loss function
In train/eval, you can choose the loss function between MSE/Huber



 

### File information
