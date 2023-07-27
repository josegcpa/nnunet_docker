import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_folder",dest="model_folder",
        help="Path to nnUNet model folder")

    args = parser.parse_args()

    data_json = json.load(
        open(os.path.join(args.model_folder,"plans.json")))

    spacing = data_json["configurations"]["3d_fullres"]["spacing"]

    print(" ".join([str(x) for x in spacing[::-1]]))