from trainer import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pd', '--pooling_date', default='max_111213', help="Pooling type + _ + experiment date in path")   # "max_111213", "avg_111213", "s2s_111213", "sum_111213"
parser.add_argument('--gpu', default=0, type=int, help="GPU number you want to use")

args = parser.parse_args()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    path = "./result/model_3DGCN/bace_cla/16_c128_d128_l2_p" + args.pd

    print(path)
    model_evaluate(path)


    # "loss": "binary_crossentropy", "monitor": "val_roc"
    # hyperparameters = {"epoch": 150, "batch": 16, "fold": 5, "units_conv": 128, "units_dense": 128, "pooling": "max",
    #                   "num_layers": 2, "loss": "mse", "monitor": "val_rmse", "label": ""}


    # conda activate tf1.15
    # cd projects/yeji/3DGCN_tf1/
    # python models/bace_eval.py --gpu 4 --pd 'avg_111213'
