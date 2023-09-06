import sys, os, argparse

# 这里的0是GPU id
import numpy as np

gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.models_pytorch import *


def main(argv):
    event_class = 24
    hidden_dim = 32
    out_dim = 64
    emb_dim = 64
    n_layers = 3

    model = MLGL(event_num=event_class,
                 hidden_dim=hidden_dim,
                 out_dim=out_dim,
                 in_dim=emb_dim,
                 n_layers=n_layers,
                 emb_dim=emb_dim, )

    from framework.pytorch_utils import count_parameters
    params_num = count_parameters(model)
    print('Parameters num: {} M'.format(params_num / 1000 ** 2))




if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















