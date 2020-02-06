from tqdm import tqdm
import json
from collections import defaultdict
from argparse import ArgumentParser
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
tf.enable_v2_behavior()


def tf2str(tf_str):
    return tf_str.numpy().decode("utf-8")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("dataset_name")
    p.add_argument("output_json")
    args = p.parse_args()

    dataset = tfds.load(args.dataset_name)
    output = defaultdict(list)
    for k in dataset.keys():
        for e in tqdm(dataset[k]):
            r = {e_k: tf2str(e_v) for (e_k, e_v) in e.items()}
            output[k].append(r)

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=4)
