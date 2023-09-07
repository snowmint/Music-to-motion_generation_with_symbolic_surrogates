from metric import compute_pck, bowing_acc
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import scipy

import torch
import torch.nn as nn

# from data import Download
import argparse
parser = argparse.ArgumentParser()
# from model.network import MovementNet
# from visualize.animation import plot

# python test.py >> test_result.txt


def calculate_fid(pred_arr, label_arr):
    '''
    calculate fid for 2-D data [time, dimension]
    input: 2d label array [time, dimension]
    2d predict array [time, dimension]
    from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
    '''

    # calculate mean and covariance statistics
    mu1, sigma1 = label_arr.mean(axis=0), np.cov(label_arr, rowvar=False)
    mu2, sigma2 = pred_arr.mean(axis=0), np.cov(pred_arr, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def main():
    # v_train = ['01', '02', '03', '04', '05']
    hidden_size_change = ['1024']
    NoY_list = ["no", "with"]
    num_epochs_change = ['300', '500']
    # ['MIDI'] #['MIDI', 'Audio', 'MIDI+Audio']
    input_data_types = ['MIDI', 'Audio', 'MIDI+Audio']
    # violin_id = ['01', '02', '03', '04', '05']
    test_pieces = ['Elgar_S1_T1', 'Elgar_S1_T2',
                   'Flower_S1_T1', 'Flower_S1_T2',
                   'Mend_S1_T1', 'Mend_S1_T2',
                   'Mozart1_S1_T1', 'Mozart1_S1_T2',
                   'Mozart2_S1_T1', 'Mozart2_S1_T2']  # test piece

    gt_data_list = []
    id = '01'
    for test_piece in test_pieces:
        # load_pickle
        gt_data_input = open("./output_eval/Ground_truth/vio" +
                             id+"_" + test_piece + "_motion_data.pkl", 'rb')
        gt_data = pickle.load(gt_data_input)
        gt_data_input.close()
        gt_data_list.append(gt_data)

    # print(len(gt_data_list), gt_data_list[0].shape)

    # MIDI data
    test_result = {}
    for data_type in input_data_types:
        test_result[data_type] = {}
        for have_annotation in NoY_list:
            for hs in hidden_size_change:
                for num_epoch in num_epochs_change:
                    this_dir = "./output_eval/"+data_type+"/"
                    filenames = next(os.walk(this_dir), (None, None, []))[2]
                    # print("filenames:", filenames)
                    # find filename include "no_anno][total300_hs1024", "total100_hs256"
                    file_code = str(have_annotation +
                                    "_anno][total"+num_epoch+"_hs"+hs)

                    test_result[data_type][file_code] = {}

                    matching = [s for s in filenames if file_code in s]
                    print("matching:", matching)

                    eval_data_input = open(this_dir+matching[0], 'rb')
                    eval_datas = pickle.load(eval_data_input)
                    eval_data_input.close()
                    # eval_datas = np.asarray(eval_datas)
                    print("eval_datas.shape", eval_datas.shape)
                    eval_datas_all = eval_datas

                    l2 = []
                    l2_hand = []
                    pck_01 = []
                    pck_02 = []
                    fd = []
                    # bow = []
                    # bowx = []
                    # bowy = []
                    # bowz = []
                    # cosine = []

                    for idx, eval_data in enumerate(eval_datas_all):
                        print(idx)
                        print("eval_data:", eval_data.shape)
                        require_len = min(
                            len(gt_data_list[idx]), len(eval_data))
                        # require_len = len(gt_data_list[idx])
                        print("require_len", require_len)
                        eval_data_cut = eval_data[:require_len].copy()
                        print(eval_data[:require_len].copy().shape)
                        print("eval_data_cut:", eval_data_cut.shape)

                        pred = eval_data_cut[:, :102]
                        targ = gt_data_list[idx][:require_len, :102].copy()
                        print("pred", pred.shape)
                        print("targ", targ.shape)
                        assert pred.shape == targ.shape

                        v_fd = calculate_fid(pred, targ)

                        # pred = pred * keypoints_std + keypoints_mean
                        pred = np.reshape(pred, [len(pred), -1, 3])
                        # targ = targ * keypoints_std + keypoints_mean
                        targ = np.reshape(targ, [len(targ), -1, 3])

                        print("pred.shape:", pred.shape)
                        print("targ.shape:", targ.shape)

                        # print("pred[:, 8:13, :]:", pred[:, 8:13, :].shape)
                        # print("targ[:, 8:13, :]:", targ[:, 8:13, :].shape)

                        # print("pred[:, 11:12, :]:", np.squeeze(pred[:, 11:12, :], axis=1).shape)
                        # print("targ[:, 11:12, :]:", np.squeeze(targ[:, 11:12, :], axis=1).shape)

                        v_l2 = np.mean(np.square(pred - targ))
                        # 8~12->right hand
                        v_l2_hand = np.mean(
                            np.square(pred[:, 8:13, :] - targ[:, 8:13, :]))
                        v_pck_01 = compute_pck(pred, targ, alpha=0.1)
                        v_pck_02 = compute_pck(pred, targ, alpha=0.2)

                        # pred_rh_wrist = np.squeeze(pred[:, 11:12, :], axis=1)
                        # targ_rh_wrist = np.squeeze(targ[:, 11:12, :], axis=1)
                        # only take right-hand wrist keypoints to calculate bowing attack accuracy
                        # v_bow_acc = bowing_acc(
                        #     pred_rh_wrist, targ_rh_wrist, alpha=3)
                        # v_cosine = np.mean(cosine_similarity(
                        #     pred_rh_wrist, targ_rh_wrist))

                        l2.append(v_l2)
                        l2_hand.append(v_l2_hand)
                        pck_01.append(v_pck_01)
                        pck_02.append(v_pck_02)
                        fd.append(v_fd)
                        # bowx.append(v_bow_acc[0])
                        # bowy.append(v_bow_acc[1])
                        # bowz.append(v_bow_acc[2])
                        # bow.append(v_bow_acc[3])
                        # cosine.append(v_cosine)
                    print("FD:", fd)
                    avg_pck = (np.mean(pck_01) + np.mean(pck_02))*0.5
                    # print(data_type + "_" + file_code + ' Avg_L1_loss: %f' %np.mean(l1))
                    # print(data_type + "_" + file_code + ' Avg_L1_hand_loss: %f' %np.mean(l1_hand))
                    # print(data_type + "_" + file_code + ' Avg_Pck: %f' %avg_pck)
                    # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracyX: %f' %np.mean(bowx))
                    # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracyY: %f' %np.mean(bowy))
                    # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracyZ: %f' %np.mean(bowz))
                    # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracy: %f' %np.mean(bow))
                    # print(data_type + "_" + file_code + ' Avg_Cosine_Similarity: %f' %np.mean(cosine))
                    test_result[data_type][file_code]["Annotation"] = have_annotation
                    test_result[data_type][file_code]["Avg_L2_loss"] = np.mean(
                        l2)
                    test_result[data_type][file_code]["Avg_L2_hand_loss"] = np.mean(
                        l2_hand)
                    test_result[data_type][file_code]["Avg_Pck"] = avg_pck
                    test_result[data_type][file_code]["Avg_FD"] = np.mean(fd)
                    # test_result[data_type][file_code]["Avg_Bowing_Attack_accuracyX"] = np.mean(
                    #     bowx)
                    # test_result[data_type][file_code]["Avg_Bowing_Attack_accuracyY"] = np.mean(
                    #     bowy)
                    # test_result[data_type][file_code]["Avg_Bowing_Attack_accuracyZ"] = np.mean(
                    #     bowz)
                    # test_result[data_type][file_code]["Avg_Bowing_Attack_accuracy"] = np.mean(
                    #     bow)
                    # test_result[data_type][file_code]["Avg_Cosine_Similarity"] = np.mean(
                    #     cosine)

    # Print Dictionary
    for data_type in input_data_types:
        print("Data type [", data_type, "]:")
        test_result_df = pd.DataFrame(test_result[data_type]).T
        test_result_df.to_csv(str('[' + data_type + ']_test_result.csv'))
        print(test_result_df)


if __name__ == '__main__':
    main()
