import sys, os, argparse

# 这里的0是GPU id
import numpy as np

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.processing import *
from framework.models_pytorch import *
from framework.utilities import cal_acc_auc
from sklearn.metrics import r2_score
from framework.data_generator import *



def cal_ar_mse_mae_r2(predictions, targets):
    mse_loss = metrics.mean_squared_error(targets, predictions)
    # rmse_loss = metrics.mean_squared_error(targets, predictions, squared=False)
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
    model = MLGL(event_num=event_class,
                 hidden_dim=hidden_dim,
                 out_dim=out_dim,
                 in_dim=emb_dim,
                 n_layers=n_layers,
                 emb_dim=emb_dim, )

    model_path = os.path.join(os.getcwd(), 'Pretrained_models', 'MLGL_attention' + config.endswith)

    model_event = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_event['state_dict'])

    if config.cuda and using_cuda:
        model.cuda()

    batch_size = 64
    generator = DataGenerator_MLGL(batch_size, emb_dim,
                                   normalization=True, )

    data_type = 'testing'
    generate_func = generator.generate_test(data_type=data_type)

    # Forward
    dict = forward_MLGL(model=model, generate_func=generate_func, cuda=using_cuda)

    acc, auc = cal_acc_auc(dict['outputs_events_l1'], dict['targets_event'])
    print("AEC 24 classes of fAEs:\n\tlevel 1 Acc: {},  AUC: {}".format(acc, auc))

    acc, auc = cal_acc_auc(dict['outputs_events_l2'], dict['targets_event'])
    print("\tlevel 2 Acc: {},  AUC: {}".format(acc, auc))

    acc, auc = cal_acc_auc(dict['outputs_events_l3'], dict['targets_event'])
    print("\tlevel 3 Acc: {},  AUC: {}".format(acc, auc))

    acc, auc = cal_acc_auc(dict['outputs_semantic7_l1'], dict['targets_semantic7'])
    print("AEC 7 classes of cAEs:\n\tlevel 1 Acc: {},  AUC: {}".format(acc, auc))

    acc, auc = cal_acc_auc(dict['outputs_semantic7_l2'], dict['targets_semantic7'])
    print("\tlevel 2 Acc: {},  AUC: {}".format(acc, auc))

    acc, auc = cal_acc_auc(dict['outputs_semantic7_l3'], dict['targets_semantic7'])
    print("\tlevel 3 Acc: {},  AUC: {}".format(acc, auc))

    mse_loss, mae_loss, r2 = cal_ar_mse_mae_r2(dict['outputs_ar_l1'], dict['target'])
    print("ARP:\n\tlevel 1 mse_loss: {},  mae_loss: {}, r2: {}".format(mse_loss, mae_loss, r2))

    mse_loss, mae_loss, r2 = cal_ar_mse_mae_r2(dict['outputs_ar_l2'], dict['target'])
    print("\tlevel 2 mse_loss: {},  mae_loss: {}, r2: {}".format(mse_loss, mae_loss, r2))

    mse_loss, mae_loss, r2 = cal_ar_mse_mae_r2(dict['outputs_ar_l3'], dict['target'])
    print("\tlevel 3 mse_loss: {},  mae_loss: {}, r2: {}".format(mse_loss, mae_loss, r2))








if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















