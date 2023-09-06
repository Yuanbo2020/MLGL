import sys, os, argparse

# 这里的0是GPU id
import numpy as np

gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *
from framework.utilities import cal_acc_auc
from sklearn.metrics import r2_score



def cal_ar_mse_mae_r2(predictions, targets):
    mse_loss = metrics.mean_squared_error(targets, predictions)
    mae_loss = metrics.mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse_loss, mae_loss, r2


def main(argv):
    event_class = 24
    hidden_dim = 32
    out_dim = 64
    emb_dim = 64
    n_layers = 3

    using_cuda = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model
    models = ['MLGL_addition', 'MLGL_concate', 'MLGL_Hadamard', 'MLGL_Gating', 'MLGL_attention']
    model_index = models.index(model_type)

    model_list = [MLGL_addition, MLGL_concate, MLGL_Hadamard, MLGL_Gating, MLGL]

    model_function = model_list[model_index]

    model = model_function(event_num=event_class, hidden_dim=hidden_dim, out_dim=out_dim, in_dim=emb_dim,
                            n_layers=n_layers, emb_dim=emb_dim, )

    model_path = os.path.join(os.getcwd(), 'Pretrained_models', model_type + config.endswith)

    model_event = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_event['state_dict'])

    if config.cuda and using_cuda:
        model.cuda()

    batch_size = 64
    generator = DataGenerator_MLGL(batch_size, emb_dim, normalization=True)

    data_type = 'testing'
    generate_func = generator.generate_test(data_type=data_type)

    # Forward
    dict = forward_MLGL_other_fusion(model=model, generate_func=generate_func, cuda=using_cuda)

    acc, auc = cal_acc_auc(dict['outputs_events'], dict['targets_event'])
    print("AEC:\n\tAcc: {},  AUC: {}".format(acc, auc))

    mse_loss, mae_loss, r2 = cal_ar_mse_mae_r2(dict['outputs_ar'], dict['target'])
    print("ARP:\n\tmse_loss: {}, mae_loss: {}, r2: {}".format(mse_loss, mae_loss, r2))






if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















