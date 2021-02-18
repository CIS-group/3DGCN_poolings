import numpy as np
import csv
import os
import sys
import argparse

from rdkit import Chem

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, NullFormatter

# import modeuls from current directory
from trainer import Trainer
from callback import calculate_roc_pr, calculate_acc
from dataset import MPGenerator


plt.rcParams['font.size'] = 16
plt.rcParams['axes.axisbelow'] = True


def load_model(trial_path):
    with open(trial_path + "/hyper.csv") as file:
        reader = csv.DictReader(file)
        for row in reader:
            hyper = dict(row)

    dataset = hyper['dataset']
    model = hyper['model']
    batch = int(hyper['batch'])
    units_conv = int(hyper['units_conv'])
    units_dense = int(hyper['units_dense'])
    num_layers = int(hyper['num_layers'])
    loss = hyper['loss']
    pooling = hyper['pooling']
    std = float(hyper['data_std'])
    mean = float(hyper['data_mean'])

    # Load model
    trainer = Trainer(dataset, split_type="stratified_sampling")
    trainer.load_data(batch=batch, iter=1)
    trainer.data.std = std
    trainer.data.mean = mean
    trainer.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers,
                       loss=loss, pooling=pooling)

    # Load best weight
    trainer.model.load_weights(trial_path + "/best_weight.hdf5")
    print("Loaded Weights from {}".format(trial_path + "/best_weight.hdf5"))

    return trainer, hyper


def random_rotation_matrix():
    theta = np.random.rand() * 2 * np.pi
    r_x = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_y = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_z = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])

    return np.matmul(np.matmul(r_x, r_y), r_z)


def degree_rotation_matrix(axis, degree):
    theta = degree / 180 * np.pi
    if axis == "x":
        r = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    elif axis == "y":
        r = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    elif axis == "z":
        r = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])
    else:
        raise ValueError("Unsupported axis for rotation: {}".format(axis))

    return r


def rotation_prediction(path, dataset, label, rotation="stepwise", axis="", degree_gap=45):
    keep_raw = []
    metric1_mean = ["acc_mean"]
    metric1_std = ["acc_std"]
    # metric2_mean = ["f1_mean"]
    # metric2_std = ["f1_std"]

    degrees = list(range(0, 360, degree_gap))
    fold = ['trial_' in filename for filename in os.listdir(path)].count(True)
    for degree in degrees:
        # Iterate over trials
        raw_results = []
        task = None
        for i in range(0, fold):
            trial_path = path + '/trial_' + str(i)
            trainer, hyper = load_model(trial_path)

            # Rotate dataset
            dataset_path = "/" + dataset + "_" + label + "_" + rotation + axis + str(degree) + ".sdf"
            if not os.path.exists(trial_path + dataset_path):
                mols = Chem.SDMolSupplier(trial_path + "/" + dataset + "_" + ".sdf")
                rotated_mols = []
                for mol in mols:
                    if label == 'active':
                        if mol.GetProp('active') == '0':
                            continue
                    elif label == 'inactive':
                        if mol.GetProp('active') == '1':
                            continue
                    if degree == 0:
                        rotated_mols.append(mol)
                        continue
                    elif rotation == "random":
                        rotation_matrix = random_rotation_matrix()
                    elif rotation == "stepwise":
                        rotation_matrix = degree_rotation_matrix(axis, float(degree))
                    else:
                        raise ValueError("Unsupported rotation mechanism: {}".format(rotation))

                    for atom in mol.GetAtoms():
                        atom_idx = atom.GetIdx()

                        pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
                        pos_rotated = np.matmul(rotation_matrix, pos)

                        mol.GetConformer().SetAtomPosition(atom_idx, pos_rotated)
                    rotated_mols.append(mol)

                # Save rotated test dataset
                w = Chem.SDWriter(trial_path + dataset_path)
                for m in rotated_mols:
                    if m is not None:
                        w.write(m)

            # Load rotation test dataset
            trainer.data.replace_dataset(trial_path + dataset_path, subset=dataset, target_name="true")

            # Predict
            if hyper["loss"] == "mse":
                computed_loss = trainer.model.evaluate_generator(trainer.data.generator(dataset))
                raw_results.append([computed_loss[1], computed_loss[2]])
            else:
                acc = calculate_acc(trainer.model, trainer.data.generator(dataset), return_pred=False)
                raw_results.append([acc])

        # Save results
        keep_raw.append(np.transpose(np.array(raw_results)))
        results_mean = np.array(raw_results).mean(axis=0)
        results_std = np.array(raw_results).std(axis=0)
        metric1_mean.append(results_mean[0])
        metric1_std.append(results_std[0])
        # metric2_mean.append(results_mean[1])
        # metric2_std.append(results_std[1])
        print(axis, degree, "finished.")

    header = [axis] + [str(degree) for degree in degrees]
    with open(path + "/" + dataset + "_" + label + "_rotation_" + axis + ".csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        writer.writerow(metric1_mean)
        writer.writerow(metric1_std)
        # writer.writerow(metric2_mean)
        # writer.writerow(metric2_std)

    header = [str(fold) + '-fold']
    with open(path + "/" + dataset + "_" + label + "_rotation_" + axis + "_raw.csv", "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        trial = 0
        degree = 0
        for r in keep_raw:
            writer.writerow([axis + str(degree) + "_acc"] + list(r[0]))
            # writer.writerow([axis + str(degree) + "_f1"] + list(r[1]))
            trial += 1
            degree += degree_gap


parser = argparse.ArgumentParser()
parser.add_argument('--pd', '--pooling_date', default='max_111213', help="Pooling type + _ + experiment date in path")
parser.add_argument('--gpu', default=0, type=int, help="GPU number you want to use")
parser.add_argument('--dataset', default='test', help="Select dataset type: train or test")
parser.add_argument('--label', default='active', help="Select data type: active or inactive")
parser.add_argument('--dg', '--degree_gap', default=10, type=int, help="Degree gap you want to rotate")
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    path = "./result/model_3DGCN/bace_cla/16_c128_d128_l2_p" + args.pd

    print(path)
    for axis in ["x", "y", "z"]:
        rotation_prediction(path, dataset=args.dataset, label=args.label, rotation="stepwise", axis=axis, degree_gap=args.dg)
