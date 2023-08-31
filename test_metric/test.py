import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import torch
import torch.nn as nn

# from data import Download
import argparse
parser = argparse.ArgumentParser()
from metric import compute_pck, bowing_acc
# from model.network import MovementNet
# from visualize.animation import plot

# python test.py >> test_result.txt

def main():
    # v_train = ['01', '02', '03', '04', '05'] 
    hidden_size_change = ['256', '1024']
    num_epochs_change = ['100', '500']
    input_data_types = ['MIDI', 'Audio', 'MIDI+Audio']#['MIDI'] #['MIDI', 'Audio', 'MIDI+Audio']
    violin_id = ['01', '02', '03', '04', '05']
    test_piece = 'Wind_S1_T2' # test piece
    
    # # Parser
    # parser = parse()
    # parser.add_argument('--plot_path', type=str, default='test.mp4', help='plot skeleton and add audio')
    # parser.add_argument('--output_path', type=str, default='test.pkl', help='save skeletal data (only for no.9 violinist)')
    # args = parser.parse_args()
    
    # # Device
    # if torch.cuda.is_available():
    #     os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # # Data
    # download_data = Download()
    # download_data.test_data()
    # with open(download_data.test_dst, 'rb') as f:
    #     Data = pickle.load(f)
    # keypoints_mean, keypoints_std = Data['keypoints_mean'], Data['keypoints_std']
    
    # # Model
    # checkpoint = torch.load(args.checkpoint, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    # movement_net = MovementNet(args.d_input, args.d_output_body, args.d_output_rh, args.d_model, args.n_block, args.n_unet, args.n_attn, args.n_head, args.max_len, args.dropout,
    #                                args.pre_lnorm, args.attn_type).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    # movement_net.load_state_dict(checkpoint['model_state_dict']['movement_net'])
    # movement_net.eval()
    gt_data_list = []
    for id in violin_id:
        # load_pickle
        gt_data_input = open("./output_eval/Ground_truth/vio"+id+"_"+ test_piece + "_motion_data.pkl", 'rb')
        gt_data = pickle.load(gt_data_input)
        gt_data_input.close()
        gt_data_list.append(gt_data)

    # print(len(gt_data_list), gt_data_list[0].shape)
    
    #MIDI data
    test_result = {}
    for data_type in input_data_types:
        test_result[data_type] = {}
        for hs in hidden_size_change:
            for num_epoch in num_epochs_change:
                this_dir = "./output_eval/"+data_type+"/"
                filenames = next(os.walk(this_dir), (None, None, []))[2]
                # print("filenames:", filenames)
                #find filename include "total100_hs256"
                file_code = str("total"+num_epoch+"_hs"+hs)

                test_result[data_type][file_code] = {}

                matching = [s for s in filenames if file_code in s]
                # print("matching:", matching)

                eval_data_input = open(this_dir+matching[0], 'rb')
                eval_datas = pickle.load(eval_data_input)
                eval_data_input.close()
                # print(eval_datas.shape)

                l2 = []
                l2_hand = []
                pck_01 = []
                pck_02 = []
                bow = []
                bowx = []
                bowy = []
                bowz = []
                cosine = []

                for idx, eval_data in enumerate(eval_datas):
                    print(idx)
                    print(eval_data.shape)
                    require_len = len(gt_data_list[idx])
                    print(require_len)
                    eval_data_cut = eval_data[:require_len]
                    print(eval_data_cut.shape)

                    pred = eval_data_cut
                    targ = gt_data_list[idx]
                    assert pred.shape==targ.shape

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
                    v_l2_hand = np.mean(np.square(pred[:, 8:13, :] - targ[:, 8:13, :])) #8~12->right hand
                    v_pck_01 = compute_pck(pred, targ, alpha=0.1)
                    v_pck_02 = compute_pck(pred, targ, alpha=0.2)
                    pred_rh_wrist = np.squeeze(pred[:, 11:12, :], axis=1)
                    targ_rh_wrist = np.squeeze(targ[:, 11:12, :], axis=1)
                    v_bow_acc = bowing_acc(pred_rh_wrist, targ_rh_wrist, alpha=3) # only take right-hand wrist keypoints to calculate bowing attack accuracy
                    v_cosine = np.mean(cosine_similarity(pred_rh_wrist, targ_rh_wrist))
        
                    l2.append(v_l2)
                    l2_hand.append(v_l2_hand)
                    pck_01.append(v_pck_01)
                    pck_02.append(v_pck_02)
                    bowx.append(v_bow_acc[0])
                    bowy.append(v_bow_acc[1])
                    bowz.append(v_bow_acc[2])
                    bow.append(v_bow_acc[3])
                    cosine.append(v_cosine)
            
                avg_pck = (np.mean(pck_01) + np.mean(pck_02))*0.5
                # print(data_type + "_" + file_code + ' Avg_L1_loss: %f' %np.mean(l1))
                # print(data_type + "_" + file_code + ' Avg_L1_hand_loss: %f' %np.mean(l1_hand))
                # print(data_type + "_" + file_code + ' Avg_Pck: %f' %avg_pck)
                # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracyX: %f' %np.mean(bowx))
                # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracyY: %f' %np.mean(bowy))
                # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracyZ: %f' %np.mean(bowz))
                # print(data_type + "_" + file_code + ' Avg_Bowing_Attack_accuracy: %f' %np.mean(bow))
                # print(data_type + "_" + file_code + ' Avg_Cosine_Similarity: %f' %np.mean(cosine))
                
                test_result[data_type][file_code]["Avg_L2_loss"] = np.mean(l2)
                test_result[data_type][file_code]["Avg_L2_hand_loss"] = np.mean(l2_hand)
                test_result[data_type][file_code]["Avg_Pck"] = avg_pck
                test_result[data_type][file_code]["Avg_Bowing_Attack_accuracyX"] = np.mean(bowx)
                test_result[data_type][file_code]["Avg_Bowing_Attack_accuracyY"] = np.mean(bowy)
                test_result[data_type][file_code]["Avg_Bowing_Attack_accuracyZ"] = np.mean(bowz)
                test_result[data_type][file_code]["Avg_Bowing_Attack_accuracy"] = np.mean(bow)
                test_result[data_type][file_code]["Avg_Cosine_Similarity"] = np.mean(cosine)
                
                
    # print("test_result:\n", test_result)
    #Print Dictionary
    for data_type in input_data_types:
        print("Data type [", data_type, "]:")
        test_result_df = pd.DataFrame(test_result[data_type]).T
        test_result_df.to_csv(str('[' + data_type + ']_test_result.csv'))
        print(test_result_df)
    # print ("{:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9}".format('Data type', 'filecode', 'Avg_L1_loss','Avg_L1_hand_loss','Avg_Pck','Avg_Bowing_Attack_accuracyX', 'Avg_Bowing_Attack_accuracyY', 'Avg_Bowing_Attack_accuracyZ', 'Avg_Bowing_Attack_accuracy', 'Avg_Cosine_Similarity'))
    # for d_t, _value in test_result.items():
    #     for f_c, value in _value:
    #         # d_t, f_c = key
    #         avg_l1, avg_l1_rh, avg_pck, avg_bow_attk_x, avg_bow_attk_y, avg_bow_attk_z, avg_bow_attk, cos_sim = value
    #         print ("{:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9}".format(d_t, f_c, avg_l1, avg_l1_rh, avg_pck, avg_bow_attk_x, avg_bow_attk_y, avg_bow_attk_z, avg_bow_attk, cos_sim ))

if __name__ == '__main__':
    main()
