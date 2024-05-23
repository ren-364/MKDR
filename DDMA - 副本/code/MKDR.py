import torch
import argparse
import numpy as np
import os.path
import dill
import time
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from models_ import MKDR
import math
from util import llprint, multi_label_metric, multi_label_metric_test, ddi_rate_score, get_n_params, buildPrjSmiles, graph_batch_from_smile


def eval(model, data_eval, voc_size, epoch, drug_data):
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1, _ = model(input[:adm_idx+1], **drug_data)
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)


            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}
        dill.dump(case_study, open(os.path.join('saved', model_name, str() + 'case_study.pkl'), 'wb'))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    ddi_rate = ddi_rate_score(smm_record, path="../ddi_A_final.pkl")

    llprint('\nddi_rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1)
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1)

def test(model, data_eval, voc_size, epoch, drug_data):
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt = 0
    visit_cnt = 0

    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for i in range(min(len(input), 5)):
            target_output1, _ = model(input[:i+1], **drug_data)
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)


            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_f1 = multi_label_metric_test(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        for i in range(min(len(input), 5)):
            ja_by_visit[i].append(adm_ja[i])
            auc_by_visit[i].append(adm_prauc[i])
            f1_by_visit[i].append(adm_f1[i])

        if step%100==0:
            print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
            print('count:', [len(buf) for buf in ja_by_visit])
            print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
            print('auc:', [np.mean(buf) for buf in auc_by_visit])
            print('f1:', [np.mean(buf) for buf in f1_by_visit])


        ja.append(adm_ja)
        prauc.append(adm_prauc)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('count:', [len(buf) for buf in ja_by_visit])
    print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
    print('auc:', [np.mean(buf) for buf in auc_by_visit])
    print('f1:', [np.mean(buf) for buf in f1_by_visit])
    result = [[len(buf) for buf in ja_by_visit], [np.mean(buf) for buf in ja_by_visit], [np.mean(buf) for buf in auc_by_visit], [np.mean(buf) for buf in f1_by_visit]]
    dill.dump(result, open(os.path.join('saved', model_name, str() + 'visit_study.pkl'), 'wb'))
    return 0


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    #数据加载
    data_path = '../records_final_demo.pkl'
    voc_path = '../voc_final.pkl'
    ehr_adj_path = '../ehr_adj_final.pkl'
    molecule_path = '../data/idx2SMILES.pkl'
    ddi_adj_path = "../ddi_A_final.pkl"
    device = torch.device('cuda:0')
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))  # diag：1958 pro：1430 med：131


    #数据集划分
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    #药物编码信息
    average_projection, smiles_list = buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)
    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': 4, 'emb_dim': 128, 'graph_pooling': 'mean',
        'drop_ratio': 0.7, 'gnn_type': 'gin', 'virtual_node': False
    }
    drug_data = {
        'mol_data': molecule_forward,
        'average_projection': average_projection
    }

    #统计先验矩阵
    def create_prior_matrices():
        Ndiag, Npro, Nmed = voc_size
        med_count_in_train = np.zeros(Nmed)
        diag_count_in_train = np.zeros(Ndiag)
        pro_count_in_train = np.zeros(Npro)
        med2med = np.zeros((Nmed, Nmed))
        diag2med = np.zeros((Ndiag, Nmed))
        pro2med = np.zeros((Npro, Nmed))
        for p in data:
            for m in p:
                cur_diag, cur_pro, cur_med, _, _, _ = m
                for cm in cur_med:
                    med2med[cm][cur_med] += 1
                    med_count_in_train[cm] += 1
                for cd in cur_diag:
                    diag2med[cd][cur_med] += 1
                    diag_count_in_train[cd] += 1
                for cp in cur_pro:
                    pro2med[cp][cur_med] += 1
                    pro_count_in_train[cp] += 1
        for cm in med_voc.idx2word:
            med2med[cm] = med2med[cm] / med_count_in_train[cm]
            med2med[cm][med2med[cm] >= 0.8] = 1
            med2med[cm][med2med[cm] < 0.8] = 0
        for cd in diag_voc.idx2word:
            diag2med[cd] = diag2med[cd] / diag_count_in_train[cd]
            #diag2med[cd][diag2med[cd] >= 0.8] = diag2med[cd]
            diag2med[cd][diag2med[cd] < 0.8] = 0
        for cp in pro_voc.idx2word:
            pro2med[cp] = pro2med[cp] / pro_count_in_train[cp]
            pro2med[cp][pro2med[cp] >= 0.8] = 1
            pro2med[cp][pro2med[cp] < 0.8] = 0
        med2med = torch.tensor(med2med).to(device)
        diag2med = torch.tensor(diag2med).to(device)
        pro2med = torch.tensor(pro2med).to(device)
        return med2med, diag2med, pro2med
    med2med, diag2med, pro2med = create_prior_matrices()

    model = MKDR(voc, voc_size, data_train, ddi_adj, global_para=molecule_para, med2med=med2med, diag2med=diag2med, ehr_adj=ehr_adj, emd_dim=emd_dim, num_heads=2, device=device)

    TEST = False
    if TEST:
        #测试
        path = '/root/data1/DDMA/code/saved/MKDR/Epoch_5_JA_0.5467.model'
        model.load_state_dict(torch.load(open(path, 'rb')))
        model.to(device=device)
        eval(model, data_eval, voc_size,1, drug_data)

    else:
        model.to(device=device)
        print('parameters', get_n_params(model))
        optimizer = Adam(list(model.parameters()), lr=LR)
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item
                    target_output1, loss_ddi = model(input=seq_input,**drug_data)

                    sigmoid_res = torch.sigmoid(target_output1)
                    loss_bce = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss_multi = F.multilabel_margin_loss(sigmoid_res, torch.LongTensor(loss3_target).to(device))

                    result = sigmoid_res.detach().cpu().numpy()[0]
                    result[result >= 0.5] = 1
                    result[result < 0.5] = 0
                    y_label = np.where(result == 1)[0]
                    current_ddi_rate = ddi_rate_score(
                        [[y_label]], path='../ddi_A_final.pkl'
                    )

                    if current_ddi_rate <= args.target_ddi:
                        loss = 0.95 * loss_bce + 0.05 * loss_multi
                    else:
                        beta = args.coef * (1 - (current_ddi_rate / args.target_ddi))
                        beta = min(math.exp(beta), 1)
                        loss = beta * (0.95 * loss_bce + 0.05 * loss_multi)  + (1 - beta) * loss_ddi


                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    #返回没有梯度更新的模块
                    for name, parms in model.named_parameters():
                        if parms.grad == None:
                            print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                                ' -->grad_value:', parms.grad)

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            ddi_rate, ja, prauc, avg_f1 = eval(model, data_eval, voc_size, epoch, drug_data)

            history['ja'].append(ja)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)
            history['ddi_rate'].append(ddi_rate)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open(os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f.model' % (epoch, ja)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja
            print(model_name)
            print(history['ja'][best_epoch], history['avg_f1'][best_epoch], history['prauc'][best_epoch], history['ddi_rate'][best_epoch])

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        print('best_epoch:', best_epoch)
        print()
        print(model_name)




if __name__ == '__main__':

    torch.manual_seed(1203)
    np.random.seed(1203)

    emd_dim = 64

    model_name = ('MKDR'
                  )
    resume_name = ''
    print('***' * 20)
    print(model_name)

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
    parser.add_argument('--model_name', type=str, default=model_name, help="model name")
    parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
    parser.add_argument(
        "--target_ddi", type=float, default=0.065, help="target ddi")
    parser.add_argument(
        '--coef', default=2.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    args = parser.parse_args()
    model_name = args.model_name
    resume_name = args.resume_path

    EPOCH = 15
    TEST = args.eval
    decay_weight = 0.85
    LR = 0.00008
    main()


