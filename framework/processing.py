import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder
from framework.models_pytorch import move_data_to_gpu
import framework.config as config
from sklearn import metrics

def define_system_name(alpha=None, basic_name='system', att_dim=None, n_heads=None,
                       batch_size=None, epochs=None):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(batch_size) + '_e' + str(epochs) \
                 + '_attd' + str(att_dim) + '_h' + str(n_heads) if att_dim is not None and n_heads is not None \
        else '_b' + str(batch_size)  + '_e' + str(epochs)

    sys_suffix = sys_suffix + '_cuda' + str(config.cuda_seed) if config.cuda_seed is not None else sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name


def forward_MLGL(model, generate_func, cuda):
    outputs_l3 = []
    outputs_events_l3 = []
    outputs_semantic7_l3 = []

    outputs_l2 = []
    outputs_events_l2 = []
    outputs_semantic7_l2 = []

    outputs_l1 = []
    outputs_events_l1 = []
    outputs_semantic7_l1 = []

    targets = []
    targets_event = []
    targets_semantic7 = []

    for data in generate_func:
        (batch_x, batch_y, batch_y_event, batch_y_semantic7, batch_graph_24, batch_graph_7, batch_graph_24_1,
         batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1) = data
        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            level1_E24_relu, level1_E7_relu, level1_ar_linear, level2_E24_relu, level2_E7_relu, level2_ar_linear, \
            level3_E24_relu, level3_E7_relu, level3_ar_linear = model(batch_x,
                                                                      batch_graph_24_1,
                                                                      batch_graph_7_1,
                                                                      batch_graph_24_7,
                                                                      batch_graph_24_7_1)
            outputs_l1.append(level1_ar_linear.data.cpu().numpy())
            outputs_events_l1.append(F.sigmoid(level1_E24_relu).data.cpu().numpy())
            outputs_semantic7_l1.append(F.sigmoid(level1_E7_relu).data.cpu().numpy())

            outputs_events_l2.append(F.sigmoid(level2_E24_relu).data.cpu().numpy())
            outputs_semantic7_l2.append(F.sigmoid(level2_E7_relu).data.cpu().numpy())
            outputs_l2.append(level2_ar_linear.data.cpu().numpy())

            outputs_events_l3.append(F.sigmoid(level3_E24_relu).data.cpu().numpy())
            outputs_semantic7_l3.append(F.sigmoid(level3_E7_relu).data.cpu().numpy())
            outputs_l3.append(level3_ar_linear.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)
            targets_semantic7.append(batch_y_semantic7)

    dict = {}

    dict['outputs_ar_l3'] = np.concatenate(outputs_l3, axis=0)
    dict['outputs_events_l3'] = np.concatenate(outputs_events_l3, axis=0)
    dict['outputs_semantic7_l3'] = np.concatenate(outputs_semantic7_l3, axis=0)

    dict['outputs_ar_l2'] = np.concatenate(outputs_l2, axis=0)

    dict['outputs_events_l2'] = np.concatenate(outputs_events_l2, axis=0)
    dict['outputs_semantic7_l2'] = np.concatenate(outputs_semantic7_l2, axis=0)

    dict['outputs_ar_l1'] = np.concatenate(outputs_l1, axis=0)
    dict['outputs_events_l1'] = np.concatenate(outputs_events_l1, axis=0)
    dict['outputs_semantic7_l1'] = np.concatenate(outputs_semantic7_l1, axis=0)

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event

    targets_semantic7 = np.concatenate(targets_semantic7, axis=0)
    dict['targets_semantic7'] = targets_semantic7
    # print(dict)
    return dict


def forward_MLGL_other_fusion(model, generate_func, cuda):
    outputs = []
    outputs_events = []

    targets = []
    targets_event = []

    for data in generate_func:
        (batch_x, batch_y, batch_y_event, batch_y_semantic7, batch_graph_24, batch_graph_7, batch_graph_24_1,
         batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1) = data
        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            _, _, _, _, _, _, E24, _, ar_linear = model(batch_x,
                                                                      batch_graph_24_1,
                                                                      batch_graph_7_1,
                                                                      batch_graph_24_7,
                                                                      batch_graph_24_7_1)

            outputs_events.append(F.sigmoid(E24).data.cpu().numpy())
            outputs.append(ar_linear.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    dict['outputs_ar'] = np.concatenate(outputs, axis=0)
    dict['outputs_events'] = np.concatenate(outputs_events, axis=0)

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict


def cal_auc(targets_event, outputs_event):
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch




def evaluate_MLGL(model, generator, data_type, cuda):

    generate_func = generator.generate_validate(data_type=data_type)

    dict = forward_MLGL(model=model, generate_func=generate_func, cuda=cuda)

    rate_mse_loss_l3 = metrics.mean_squared_error(dict['target'], dict['outputs_ar_l3'])
    auc_event_l3 = cal_auc(dict['targets_event'], dict['outputs_events_l3'])
    auc_semantic7_l3 = cal_auc(dict['targets_semantic7'], dict['outputs_semantic7_l3'])

    rate_mse_loss_l2 = metrics.mean_squared_error(dict['target'], dict['outputs_ar_l2'])
    auc_event_l2 = cal_auc(dict['targets_event'], dict['outputs_events_l2'])
    auc_semantic7_l2 = cal_auc(dict['targets_semantic7'], dict['outputs_semantic7_l2'])

    rate_mse_loss_l1 = metrics.mean_squared_error(dict['target'], dict['outputs_ar_l1'])
    auc_event_l1 = cal_auc(dict['targets_event'], dict['outputs_events_l1'])
    auc_semantic7_l1 = cal_auc(dict['targets_semantic7'], dict['outputs_semantic7_l1'])

    return rate_mse_loss_l3, auc_event_l3, auc_semantic7_l3, \
           rate_mse_loss_l2, auc_event_l2, auc_semantic7_l2, \
           rate_mse_loss_l1, auc_event_l1, auc_semantic7_l1




def training_graph_with_semantic7(generator, model, cuda, models_dir, public_modelpath, epochs, adamw,
             batch_size, save_every_epoch, lr_init=1e-3,
             log_path=None, check_step=1, alpha=None):
    create_folder(models_dir)

    # Optimizer
    if adamw:
        optimizer = optim.AdamW(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    min_val_mse_ar_l3 = 100
    min_val_mse_ar_l3_itera = 0
    val_mse_ar_l3 = []
    val_mse_ar_l3_file = os.path.join(log_path, 'val_mse_ar_l3.txt')

    max_val_auc_event_l3 = 0.000001
    max_val_auc_event_l3_itera = 0
    val_auc_event_l3 = []
    val_auc_event_l3_file = os.path.join(log_path, 'val_auc_event_l3.txt')

    max_val_auc_semantic7_l3 = 0.000001
    max_val_auc_semantic7_l3_itera = 0
    val_auc_semantic7_l3 = []
    val_auc_semantic7_l3_file = os.path.join(log_path, 'val_auc_semantic7_l3.txt')

    # ------------------------------------------------------------------------------
    min_val_mse_ar_l2 = 100
    min_val_mse_ar_l2_itera = 0
    val_mse_ar_l2 = []
    val_mse_ar_l2_file = os.path.join(log_path, 'val_mse_ar_l2.txt')

    max_val_auc_event_l2 = 0.000001
    max_val_auc_event_l2_itera = 0
    val_auc_event_l2 = []
    val_auc_event_l2_file = os.path.join(log_path, 'val_auc_event_l2.txt')

    max_val_auc_semantic7_l2 = 0.000001
    max_val_auc_semantic7_l2_itera = 0
    val_auc_semantic7_l2 = []
    val_auc_semantic7_l2_file = os.path.join(log_path, 'val_auc_semantic7_l2.txt')

    # ------------------------------------------------------------------------------
    min_val_mse_ar_l1 = 100
    min_val_mse_ar_l1_itera = 0
    val_mse_ar_l1 = []
    val_mse_ar_l1_file = os.path.join(log_path, 'val_mse_ar_l1.txt')

    max_val_auc_event_l1 = 0.000001
    max_val_auc_event_l1_itera = 0
    val_auc_event_l1 = []
    val_auc_event_l1_file = os.path.join(log_path, 'val_auc_event_l1.txt')

    max_val_auc_semantic7_l1 = 0.000001
    max_val_auc_semantic7_l1_itera = 0
    val_auc_semantic7_l1 = []
    val_auc_semantic7_l1_file = os.path.join(log_path, 'val_auc_semantic7_l1.txt')

    save_best_rate_model_l3 = 0
    save_best_rate_model_l2 = 0
    save_best_rate_model_l1 = 0
    save_best_event_model_l3 = 0
    save_best_event_model_l2 = 0
    save_best_event_model_l1 = 0

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    sample_num = len(generator.train_audio_ids)
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'iteration is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = int(one_epoch / check_step)
    print('validating every: ', check_iter, ' iteration')

    # Train on mini batches
    for iteration, all_data in enumerate(generator.generate_train()):

        (batch_x, batch_y_cpu, batch_y_event_cpu, batch_y_semantic7, batch_graph_24, batch_graph_7, batch_graph_24_1,
          batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1) = all_data
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y_rate = move_data_to_gpu(batch_y_cpu, cuda)
        batch_y_event = move_data_to_gpu(batch_y_event_cpu, cuda)
        batch_y_semantic7 = move_data_to_gpu(batch_y_semantic7, cuda)

        model.train()
        optimizer.zero_grad()

        level1_E24_relu, level1_E7_relu, level1_ar_linear, level2_E24_relu, level2_E7_relu, level2_ar_linear, \
        level3_E24_relu, level3_E7_relu, level3_ar_linear = model(batch_x,
                                                                                       batch_graph_24_1,
                                                                                       batch_graph_7_1,
                                                                                       batch_graph_24_7,
                                                                                       batch_graph_24_7_1)

        # print(linear_each_events.shape, linear_semantic7.shape, linear_rate.shape, linear_each_events_level3.shape,
        #       linear_semantic7_level3.shape, linear_each_events_level2.shape, linear_semantic7_level2.shape,
        #       linear_rate_level2.shape)
        # print(batch_y_event.shape, batch_y_semantic7.shape, batch_y_rate.shape)
        # # torch.Size([64, 24]) torch.Size([64, 7]) torch.Size([64])
        # # torch.Size([64, 24]) torch.Size([64, 7]) torch.Size([64, 1])

        level1_loss_each_event24 = bce_loss(F.sigmoid(level1_E24_relu), batch_y_event)
        level1_loss_semantic7 = bce_loss(F.sigmoid(level1_E7_relu), batch_y_semantic7)
        level1_loss_ar = mse_loss(level1_ar_linear, batch_y_rate)

        # ---------------------- level 2 loss --------------------------------------------------------------------
        level2_loss_each_event24 = bce_loss(F.sigmoid(level2_E24_relu), batch_y_event)
        level2_loss_semantic7 = bce_loss(F.sigmoid(level2_E7_relu), batch_y_semantic7)
        level2_loss_ar = mse_loss(level2_ar_linear, batch_y_rate)

        # ---------------------- level 3 loss --------------------------------------------------------------------
        level3_loss_each_event24 = bce_loss(F.sigmoid(level3_E24_relu), batch_y_event)
        level3_loss_semantic7 = bce_loss(F.sigmoid(level3_E7_relu), batch_y_semantic7)
        level3_loss_ar = mse_loss(level3_ar_linear, batch_y_rate)

        if alpha is not None:
            if type(alpha[0]) == str:
                alpha = [float(each) for each in alpha]
                loss_common = alpha[0] * level1_loss_each_event24 + alpha[1] * level1_loss_semantic7 + alpha[2] * level1_loss_ar \
                              + alpha[3] * level2_loss_each_event24 + alpha[4] * level2_loss_semantic7 + alpha[5] * level2_loss_ar \
                              + alpha[6] * level3_loss_each_event24 + alpha[7] * level3_loss_semantic7 + alpha[8] * level3_loss_ar
            else:
                loss_common = alpha[0] * level1_loss_each_event24 + alpha[1] * level1_loss_semantic7 + alpha[
                    2] * level1_loss_ar \
                              + alpha[3] * level2_loss_each_event24 + alpha[4] * level2_loss_semantic7 + alpha[
                                  5] * level2_loss_ar \
                              + alpha[6] * level3_loss_each_event24 + alpha[7] * level3_loss_semantic7 + alpha[
                                  8] * level3_loss_ar
        else:
            loss_common = level1_loss_each_event24 + level1_loss_semantic7 + level1_loss_ar \
                          + level2_loss_each_event24 + level2_loss_semantic7 + level2_loss_ar \
                          + level3_loss_each_event24 + level3_loss_semantic7 + level3_loss_ar

        loss_common.backward()
        optimizer.step()

        if alpha is None:
            print('epoch: ', '%.4f' % (iteration / one_epoch), 'loss: %.5f' % float(loss_common),
                  'l3_AR: %.5f' % float(level3_loss_ar),
                  'l3_e24: %.5f' % float(level3_loss_semantic7),
                  'l3_e7: %.5f' % float(level3_loss_each_event24),

                  'l2_AR: %.5f' % float(level2_loss_ar),
                  'l2_e24: %.5f' % float( level2_loss_semantic7),
                  'l2_e7: %.5f' % float(level2_loss_each_event24),

                  'l1_AR: %.5f' % float(level1_loss_ar),
                  'l1_e24: %.5f' % float(level1_loss_semantic7),
                  'l1_e7: %.5f' % float(level1_loss_each_event24),
                  )
        else:
            print('epoch: ', '%.4f' % (iteration / one_epoch), 'loss: %.5f' % float(loss_common),
                  'l3_AR: %.5f' % float(alpha[8] * level3_loss_ar),
                  'l3_e24: %.5f' % float(alpha[7] * level3_loss_semantic7),
                  'l3_e7: %.5f' % float(alpha[6] * level3_loss_each_event24),

                  'l2_AR: %.5f' % float(alpha[5] * level2_loss_ar),
                  'l2_e24: %.5f' % float(alpha[4] * level2_loss_semantic7),
                  'l2_e7: %.5f' % float(alpha[3] * level2_loss_each_event24),

                  'l1_AR: %.5f' % float(alpha[2] * level1_loss_ar),
                  'l1_e24: %.5f' % float(alpha[1] * level1_loss_semantic7),
                  'l1_e7: %.5f' % float(alpha[0] * level1_loss_each_event24),
                  )


        # 6122 / 64 = 95.656
        if iteration % check_iter == 0 and iteration > 0:
            train_fin_time = time.time()
            rate_mse_loss_l3, auc_event_l3, auc_semantic7_l3, \
            rate_mse_loss_l2, auc_event_l2, auc_semantic7_l2, \
            rate_mse_loss_l1, auc_event_l1, auc_semantic7_l1 = evaluate_asc_aec(model=model,
                                                              generator=generator,
                                                              data_type='validate',
                                                              cuda=cuda)

            val_mse_ar_l3.append(rate_mse_loss_l3)
            val_auc_event_l3.append(auc_event_l3)
            val_auc_semantic7_l3.append(auc_semantic7_l3)

            val_mse_ar_l2.append(rate_mse_loss_l2)
            val_auc_event_l2.append(auc_event_l2)
            val_auc_semantic7_l2.append(auc_semantic7_l2)

            val_mse_ar_l1.append(rate_mse_loss_l1)
            val_auc_event_l1.append(auc_event_l1)
            val_auc_semantic7_l1.append(auc_semantic7_l1)

            # -----------------------------------------------------------------------------
            if auc_event_l3 > max_val_auc_event_l3:
                save_best_event_model_l3 = 1
                max_val_auc_event_l3 = auc_event_l3
                max_val_auc_event_l3_itera = iteration / one_epoch

            if auc_semantic7_l3 > max_val_auc_semantic7_l3:
                max_val_auc_semantic7_l3 = auc_semantic7_l3
                max_val_auc_semantic7_l3_itera = iteration / one_epoch

            if rate_mse_loss_l3 < min_val_mse_ar_l3:
                save_best_rate_model_l3 = 1
                min_val_mse_ar_l3 = rate_mse_loss_l3
                min_val_mse_ar_l3_itera = iteration / one_epoch
            # -----------------------------------------------------------------------------
            if auc_event_l2 > max_val_auc_event_l2:
                save_best_event_model_l2 = 1
                max_val_auc_event_l2 = auc_event_l2
                max_val_auc_event_l2_itera = iteration / one_epoch

            if auc_semantic7_l2 > max_val_auc_semantic7_l2:
                max_val_auc_semantic7_l2 = auc_semantic7_l2
                max_val_auc_semantic7_l2_itera = iteration / one_epoch

            if rate_mse_loss_l2 < min_val_mse_ar_l2:
                save_best_rate_model_l2 = 1
                min_val_mse_ar_l2 = rate_mse_loss_l2
                min_val_mse_ar_l2_itera = iteration / one_epoch
            # -----------------------------------------------------------------------------
            if auc_event_l1 > max_val_auc_event_l1:
                save_best_event_model_l1 = 1
                max_val_auc_event_l1 = auc_event_l1
                max_val_auc_event_l1_itera = iteration / one_epoch

            if auc_semantic7_l1 > max_val_auc_semantic7_l1:
                max_val_auc_semantic7_l1 = auc_semantic7_l1
                max_val_auc_semantic7_l1_itera = iteration / one_epoch

            if rate_mse_loss_l1 < min_val_mse_ar_l1:
                save_best_rate_model_l1 = 1
                min_val_mse_ar_l1 = rate_mse_loss_l1
                min_val_mse_ar_l1_itera = iteration / one_epoch
            # -----------------------------------------------------------------------------

            validate_time = time.time() - train_fin_time

            print('E: ', '%.4f' % (iteration / one_epoch), ' l3_ar: %.3f' % rate_mse_loss_l3,
                  ' l3_e24: %.3f' % auc_event_l3, ' l3_e7: %.3f' % auc_semantic7_l3,

                  ' l2_ar: %.3f' % rate_mse_loss_l2,
                  ' l2_e24: %.3f' % auc_event_l2, ' l2_e7: %.3f' % auc_semantic7_l2,

                  ' l1_ar: %.3f' % rate_mse_loss_l1,
                  ' l1_e24: %.3f' % auc_event_l1, ' l1_e7: %.3f' % auc_semantic7_l1,
                  )


            if rate_mse_loss_l3 < 1.0 and auc_event_l3 > 0.920:
                save_out_dict = {'state_dict': model.state_dict()}
                model_name = 'md_e' + str(iteration / one_epoch) + \
                                             '_3e24_' + '%.3f' % auc_event_l3 +\
                                             '_3ar_' + '%.3f' % rate_mse_loss_l3 +\
                                             '_2e24_' + '%.3f' % auc_event_l2 +\
                                             '_2ar_' + '%.3f' % rate_mse_loss_l2 +\
                                             '_1e24_' + '%.3f' % auc_event_l1 +\
                                             '_1ar_' + '%.3f' % rate_mse_loss_l1
                save_out_path = os.path.join(public_modelpath, model_name + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Model saved to {}'.format(save_out_path))
                source_txt = os.path.join(public_modelpath, model_name + '.txt')
                with open(source_txt, 'w') as f:
                    f.write(models_dir)

            # if iteration % (one_epoch * save_every_epoch) == 0 and iteration > 0:
            #     save_out_dict = {'state_dict': model.state_dict()}
            #     model_name = 'md_e' + str(iteration / one_epoch) + \
            #                  '_3e24_' + '%.3f' % auc_event_l3 + \
            #                  '_3ar_' + '%.3f' % rate_mse_loss_l3 + \
            #                  '_2e24_' + '%.3f' % auc_event_l2 + \
            #                  '_2ar_' + '%.3f' % rate_mse_loss_l2 + \
            #                  '_1e24_' + '%.3f' % auc_event_l1 + \
            #                  '_1ar_' + '%.3f' % rate_mse_loss_l1
            #     save_out_path = os.path.join(models_dir, model_name + config.endswith)
            #     torch.save(save_out_dict, save_out_path)

            print('E: {}, T_val: {:.3f} s, '
                  'min_l3_ar: {:.3f} , ite: {:.2f},  '
                  '  max_l3_e24: {:.3f} , ite: {:.2f} , '
                  '  max_l3_e7: {:.3f} , ite: {:.2f} ,'
                  
                  'min_l2_ar: {:.3f} , ite: {:.2f},  '
                  '  max_l2_e24: {:.3f} , ite: {:.2f} , '
                  '  max_l2_e7: {:.3f} , ite: {:.2f} ,'

                  'min_l1_ar: {:.3f} , ite: {:.2f},  '
                  '  max_l1_e24: {:.3f} , ite: {:.2f} , '
                  '  max_l1_e7: {:.3f} , ite: {:.2f} ,'
                  .format('%.4f' % (iteration / one_epoch), validate_time,
                          min_val_mse_ar_l3, min_val_mse_ar_l3_itera,
                          max_val_auc_event_l3, max_val_auc_event_l3_itera,
                          max_val_auc_semantic7_l3, max_val_auc_semantic7_l3_itera,

                          min_val_mse_ar_l2, min_val_mse_ar_l2_itera,
                          max_val_auc_event_l2, max_val_auc_event_l2_itera,
                          max_val_auc_semantic7_l2, max_val_auc_semantic7_l2_itera,

                          min_val_mse_ar_l1, min_val_mse_ar_l1_itera,
                          max_val_auc_event_l1, max_val_auc_event_l1_itera,
                          max_val_auc_semantic7_l1, max_val_auc_semantic7_l1_itera,
                          ))

            np.savetxt(val_mse_ar_l3_file, val_mse_ar_l3, fmt='%.5f')
            np.savetxt(val_auc_event_l3_file, val_auc_event_l3, fmt='%.5f')
            np.savetxt(val_auc_semantic7_l3_file, val_auc_semantic7_l3, fmt='%.5f')

            np.savetxt(val_mse_ar_l2_file, val_mse_ar_l2, fmt='%.5f')
            np.savetxt(val_auc_event_l2_file, val_auc_event_l2, fmt='%.5f')
            np.savetxt(val_auc_semantic7_l2_file, val_auc_semantic7_l2, fmt='%.5f')

            np.savetxt(val_mse_ar_l1_file, val_mse_ar_l1, fmt='%.5f')
            np.savetxt(val_auc_event_l1_file, val_auc_event_l1, fmt='%.5f')
            np.savetxt(val_auc_semantic7_l1_file, val_auc_semantic7_l1, fmt='%.5f')

        if save_best_rate_model_l3:
            save_best_rate_model_l3 = 0
            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'rate_best_l3' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Best rate model saved to {}'.format(save_out_path))

        if save_best_rate_model_l2:
            save_best_rate_model_l2 = 0
            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'rate_best_l2' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Best rate model saved to {}'.format(save_out_path))

        if save_best_rate_model_l1:
            save_best_rate_model_l1 = 0
            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'rate_best_l1' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Best rate model saved to {}'.format(save_out_path))

        if save_best_event_model_l3:
            save_best_event_model_l3 = 0
            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'event_best_l3' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Best event model saved to {}'.format(save_out_path))

        if save_best_event_model_l2:
            save_best_event_model_l2 = 0
            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'event_best_l2' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Best event model saved to {}'.format(save_out_path))

        if save_best_event_model_l1:
            save_best_event_model_l1 = 0
            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'event_best_l1' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Best event model saved to {}'.format(save_out_path))

        # Stop learning
        if iteration > (epochs * one_epoch):

            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'final_e' +str(epochs) + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Final model saved to {}'.format(save_out_path))

            print('E: {}, T_val: {:.3f} s, '
                  'min_l3_ar: {:.3f} , ite: {:.2f},  '
                  '  max_l3_e24: {:.3f} , ite: {:.2f} , '
                  '  max_l3_e7: {:.3f} , ite: {:.2f} ,'

                  'min_l2_ar: {:.3f} , ite: {:.2f},  '
                  '  max_l2_e24: {:.3f} , ite: {:.2f} , '
                  '  max_l2_e7: {:.3f} , ite: {:.2f} ,'

                  'min_l1_ar: {:.3f} , ite: {:.2f},  '
                  '  max_l1_e24: {:.3f} , ite: {:.2f} , '
                  '  max_l1_e7: {:.3f} , ite: {:.2f} ,'
                  .format('%.4f' % (iteration / one_epoch), validate_time,
                          min_val_mse_ar_l3, min_val_mse_ar_l3_itera,
                          max_val_auc_event_l3, max_val_auc_event_l3_itera,
                          max_val_auc_semantic7_l3, max_val_auc_semantic7_l3_itera,

                          min_val_mse_ar_l2, min_val_mse_ar_l2_itera,
                          max_val_auc_event_l2, max_val_auc_event_l2_itera,
                          max_val_auc_semantic7_l2, max_val_auc_semantic7_l2_itera,

                          min_val_mse_ar_l1, min_val_mse_ar_l1_itera,
                          max_val_auc_event_l1, max_val_auc_event_l1_itera,
                          max_val_auc_semantic7_l1, max_val_auc_semantic7_l1_itera,
                          ))

            np.savetxt(val_mse_ar_l3_file, val_mse_ar_l3, fmt='%.5f')
            np.savetxt(val_auc_event_l3_file, val_auc_event_l3, fmt='%.5f')
            np.savetxt(val_auc_semantic7_l3_file, val_auc_semantic7_l3, fmt='%.5f')

            np.savetxt(val_mse_ar_l2_file, val_mse_ar_l2, fmt='%.5f')
            np.savetxt(val_auc_event_l2_file, val_auc_event_l2, fmt='%.5f')
            np.savetxt(val_auc_semantic7_l2_file, val_auc_semantic7_l2, fmt='%.5f')

            np.savetxt(val_mse_ar_l1_file, val_mse_ar_l1, fmt='%.5f')
            np.savetxt(val_auc_event_l1_file, val_auc_event_l1, fmt='%.5f')
            np.savetxt(val_auc_semantic7_l1_file, val_auc_semantic7_l1, fmt='%.5f')

            print('Training is done!!!')
            break






