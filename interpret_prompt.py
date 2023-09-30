import os
import json
import torch

from utils.tokenizer import SimpleTokenizer
from models.ULIP_models import ULIP_PointBERT


def main(args):
    fpath = args.fpath
    topk = args.topk

    assert os.path.exists(fpath)
    print(f"Return the top-{topk} matched words")

    with open('data/labels.json') as fin:
        labels = json.load(fin)
        args.classnames = labels[args.dataset_name]

    tokenizer = SimpleTokenizer()
    model = ULIP_PointBERT(args)
    token_embedding = model.token_embedding.weight
    print(f"Size of token embedding: {token_embedding.shape}")

    prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
    # check the keys in `prompt_learner`, actually, the keys are the names of learnable parameters
    print('prompt_learner.keys():', prompt_learner.keys())
    ctx = prompt_learner["learnable_tokens"]
    ctx = ctx.float()
    print(f"Size of context: {ctx.shape}")

    if ctx.dim() == 2:
        # Generic context
        distance = torch.cdist(ctx, token_embedding)
        print(f"Size of distance matrix: {distance.shape}")
        sorted_idxs = torch.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :topk]

        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.decoder[idx.item()] for idx in idxs]
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
            print(f"{m+1}: {words} {dist}")


if __name__ == '__main__':
    from parser import args
    
    main(args)