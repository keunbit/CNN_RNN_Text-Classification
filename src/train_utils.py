import os
import re

import torch
import yaml


def get_ckpt_version(args):
    version_l = [int(f_.split("_")[-1]) for f_ in os.listdir(args.output_dir) if re.match(r"version_[0-9]+$", f_)]
    if len(version_l) < 1:
        os.makedirs(f"{args.output_dir}/version_0")
        return 0
    cur_version = max(version_l) + 1
    os.makedirs(f"{args.output_dir}/version_{cur_version}")
    return cur_version


def save_model(args, ckpt_version, ckpt_model):
    torch.save(ckpt_model.state_dict(), f"{args.output_dir}/version_{ckpt_version}/nsmc_cnn_model.ckpt")
    with open(f"{args.output_dir}/version_{ckpt_version}/hparams.yaml", "w") as f:
        yaml.dump(args, f)


def save_score(args, ckpt_version, step, score, cur_epochs):
    if not os.path.exists(f"{args.output_dir}/version_{ckpt_version}/eval_history.txt"):
        with open(f"{args.output_dir}/version_{ckpt_version}/eval_history.txt", "w") as f:
            f.write(f"train_{step+1} step - accuracy : {score}% - epochs : {cur_epochs}\n")
    else:
        with open(f"{args.output_dir}/version_{ckpt_version}/eval_history.txt", "a") as f:
            f.write(f"train_{step+1} step - accuracy : {score}% - epochs : {cur_epochs}\n")
