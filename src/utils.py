import os
import torch


def load_model(args, save_file):
    path = os.path.join(
        args.model_dir,
        args.model_type,
        save_file,
        "checkpoint-{}".format(args.max_steps),
        "model.pt",
    )
    assert os.path.exists(path), "Path is not exist"
    model = torch.load(path)
    return model, path
