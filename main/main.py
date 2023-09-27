
from run.cross_validation import cross_valid
from run.run import run
from run.prediction import predict
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--kfold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--kfold_cv_type", type=str, default='pair', choices=['pair', 'drug', 'cellline'], help="K-Fold Cross-Validation dataset splitting choice")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss") 
    parser.add_argument("--n_epochs", type=int, default=512, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="model batch size")
    parser.add_argument("--task", type=str, default='cv', choices=['cv', 'run', 'predict'], help="model work mode, cross-validating, running or predicting")
    parser.add_argument("--monitor", type=str, default='loss_val', choices=['loss_val', 'acc_val'], help="earlystop monitor")
    parser.add_argument("--metric", type=str, default='ACC', choices=['ACC', 'ROC'], help="optimaizer metric")
    parser.add_argument("--source", type=str, default='GDSC,CTRP,CCLE', help="the data source that the task working on")
    parser.add_argument("--scale_method", type=str, default='standard', choices=['standard', 'minmax', 'None'], help="")



    params = parser.parse_args()
    print(vars(params))

    if params.task=='cv':
        cver = cross_valid(params)
        cver.run()
    elif params.task=='run':
        runner = run(params)
        runner.run()
    elif params.task=='predict':
        predictner = predict(params)
        predictner.run()