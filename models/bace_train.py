from trainer import Trainer
import os
import argparse

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--pooling', default='avg', help="Pooling type")
parser.add_argument('--gpu', default=0, help="GPU number you want to use")

parser.add_argument('--fold', default=10, type=int, help="k-fold cross validation")

args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    trainer = Trainer("bace_cla", split_type="stratified_sampling")
    hyperparameters = {"epoch": 150, "batch": 8, "fold": args.fold, "units_conv": 128, "units_dense": 128, "pooling": args.pooling,
                       "num_layers": 2, "loss": "binary_crossentropy", "monitor": "val_roc", "label": ""}

    features = {"use_atom_symbol": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_ring_size": True, "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}

    # Baseline
    trainer.fit("model_3DGCN", use_multiprocessing=False, **hyperparameters, **features)

