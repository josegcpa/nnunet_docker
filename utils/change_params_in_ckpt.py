"""
Command line utility to change parameter names in a torch checkpoint.
"""

import torch

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Changes parameters in a torch checkpoint")
    parser.add_argument(
        "--checkpoints",
        type=str,
        help="Path to checkpoint file",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--new_params",
        type=str,
        nargs="+",
        help="Name of parameter to change",
        required=True,
    )
    parser.add_argument(
        "--out_paths",
        type=str,
        default=None,
        help="Path to output checkpoint file (inplace if not specified)",
    )

    args = parser.parse_args()

    new_params = {}

    for kv in args.new_params:
        k, v = kv.split("=")
        k = tuple(k.split("."))
        v = eval(v) if v[0] != '"' else v
        new_params[k] = v

    for i, ckpt_path in enumerate(args.checkpoints):
        ckpt = torch.load(ckpt_path)
        for k in new_params:
            cmd = (
                "ckpt"
                + "".join([f'["{kk}"]' for kk in k])
                + f" = {new_params[k]}"
            )
            exec(cmd)
        if args.out_paths is None:
            torch.save(ckpt, ckpt_path)
        else:
            torch.save(ckpt, args.out_paths)
