import pretty_midi
import numpy as np
import glob
import pandas as pd
import scipy
from scipy import signal
import pickle
import os
# import pprint

motion_sf = 250
change_fps = 27


def flatten(l):
    return [item for sublist in l for item in sublist]


def read_midi(filename, specific_fps):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(filename)

    piano_roll = midi_data.get_piano_roll(fs=specific_fps)  # 40fps #250fps
    piano_roll[piano_roll > 0] = 1

    return piano_roll


# this_performer = "vio01"
performaer_list = ["vio01", "vio02", "vio03", "vio04", "vio05"]
# performaer_list = ["vio01", "vio02", "vio03", "vio04", "vio05"]
piece_list = ["Bach1_S1_T1", "Bach1_S1_T2", "Bach2_S1_T1", "Bach2_S1_T2", "Bach2_S2_T1", "Bach2_S2_T2", "Beeth1_S1_T1", "Beeth1_S1_T2", "Beeth2_S1_T1", "Beeth2_S1_T2", "Elgar_S1_T1",
              "Elgar_S1_T2", "Flower_S1_T1", "Flower_S1_T2", "Mend_S1_T1", "Mend_S1_T2", "Mozart1_S1_T1", "Mozart1_S1_T2", "Mozart2_S1_T1", "Mozart2_S1_T2", "Wind_S1_T1", "Wind_S1_T2"]

# Read Motion
keypoint_dictionary = {}
for this_performer in performaer_list:
    keypoint_dictionary[this_performer] = {}

training_dataset = {}  # 'aud', 'keypoints', 'seq_len'=900
UoE_mocap_path = "./UoE_violin_midi/performance_motion/"

body_joint_dict = {"RFHD_X": 0, "RFHD_Y": 1, "RFHD_Z": 2,  # head
                   "LFHD_X": 3, "LFHD_Y": 4, "LFHD_Z": 5,
                   "RBHD_X": 6, "RBHD_Y": 7, "RBHD_Z": 8,
                   "LBHD_X": 9, "LBHD_Y": 10, "LBHD_Z": 11,
                   "C7_X": 12, "C7_Y": 13, "C7_Z": 14,
                   "T10_X": 15, "T10_Y": 16, "T10_Z": 17,
                   "CLAV_X": 18, "CLAV_Y": 19, "CLAV_Z": 20,
                   "STRN_X": 21, "STRN_Y": 22, "STRN_Z": 23,
                   "RSHO_X": 24, "RSHO_Y": 25, "RSHO_Z": 26,
                   "RELB_X": 27, "RELB_Y": 28, "RELB_Z": 29,
                   "RWRA_X": 30, "RWRA_Y": 31, "RWRA_Z": 32,
                   "RWRB_X": 33, "RWRB_Y": 34, "RWRB_Z": 35,
                   "RFIN_X": 36, "RFIN_Y": 37, "RFIN_Z": 38,
                   "LSHO_X": 39, "LSHO_Y": 40, "LSHO_Z": 41,
                   "LELB_X": 42, "LELB_Y": 43, "LELB_Z": 44,
                   "LWRA_X": 45, "LWRA_Y": 46, "LWRA_Z": 47,
                   "LWRB_X": 48, "LWRB_Y": 49, "LWRB_Z": 50,
                   "LFIN_X": 51, "LFIN_Y": 52, "LFIN_Z": 53,
                   # (18*3) left waist
                   "RASI_X": 54, "RASI_Y": 55, "RASI_Z": 56,
                   # (19*3) right waist
                   "LASI_X": 57, "LASI_Y": 58, "LASI_Z": 59,
                   "RPSI_X": 60, "RPSI_Y": 61, "RPSI_Z": 62,
                   "LPSI_X": 63, "LPSI_Y": 64, "LPSI_Z": 65,
                   "RKNE_X": 66, "RKNE_Y": 67, "RKNE_Z": 68,
                   "RHEE_X": 69, "RHEE_Y": 70, "RHEE_Z": 71,
                   "RTOE_X": 72, "RTOE_Y": 73, "RTOE_Z": 74,
                   "RANK_X": 75, "RANK_Y": 76, "RANK_Z": 77,
                   "LKNE_X": 78, "LKNE_Y": 79, "LKNE_Z": 80,
                   "LHEE_X": 81, "LHEE_Y": 82, "LHEE_Z": 83,
                   "LTOE_X": 84, "LTOE_Y": 85, "LTOE_Z": 86,
                   "LANK_X": 87, "LANK_Y": 88, "LANK_Z": 89,  # 29*3
                   "Rightant_X": 90, "Rightant_Y": 91, "Rightant_Z": 92,
                   "Rightpost_X": 93, "Rightpost_Y": 94, "Rightpost_Z": 95,
                   "Leftant_X": 96, "Leftant_Y": 97, "Leftant_Z": 98,
                   "Leftpost_X": 99, "Leftpost_Y": 100, "Leftpost_Z": 101}

shape_dict = {}

shape_dict['selected_frame_shape'] = []
shape_dict['combined_shape'] = []
keypoint_list = []
target_dict = {}

for this_performer in performaer_list:
    for song_name in piece_list:
        file_full_path = UoE_mocap_path + this_performer + \
            "/" + this_performer + "_" + song_name + ".csv"
        print("===============================")
        print(file_full_path)
        splited_name = str(file_full_path).split('/')
        name_code = splited_name[4].split('.')[0]
        print("\nthis performer: ", this_performer)
        print("\nname_code:", name_code)

        mocap_metadata = pd.read_csv(file_full_path, nrows=5, names=[
                                     'attr', 'value'], on_bad_lines='skip', header=None)
        mocap_data = pd.read_csv(
            file_full_path, skiprows=5, on_bad_lines='skip')

        mocap_n_frame = mocap_metadata.loc[mocap_metadata['attr']
                                           == "NO_OF_FRAMES", "value"].item()
        mocap_fps = mocap_metadata.loc[mocap_metadata['attr']
                                       == "SAMPLING FREQUENCY (FPS)", "value"].item()
        mocap_n_frame = int(mocap_n_frame)
        mocap_fps = int(mocap_fps)
        print("mocap_n_frame", mocap_n_frame)
        print("mocap_fps", mocap_fps)
        print(mocap_data.shape)

        # Fill NAN
        mocap_data = mocap_data[(mocap_data.T != 0).any()]
        mocap_data = mocap_data.replace(0.0, np.NaN)
        mocap_data = mocap_data.interpolate(axis=0)
        mocap_data = mocap_data.bfill(axis=0)
        mocap_data = mocap_data.ffill(axis=0)

        # Downsampling to specific fps
        new_length = int(round((mocap_n_frame/250)*change_fps, 0))
        resampled = signal.resample(mocap_data, new_length)

        mocap_data_downsample = pd.DataFrame(
            resampled, columns=mocap_data.columns)
        print("mocap_data_downsample.shape:", mocap_data_downsample.shape)
        print("---")

        """
        #針對 joint 進行正規化
        normalization: 把每一個關節點設為原點: ex 左腳腳底(比較不會動 point 30 作為原點)，

        Naive:
        point 1 的 y 軸：假設人的身高是 1 (xxx, 1, zzz)
        """
        x_cols = [col for col in mocap_data_downsample.columns if '_X' in col]
        y_cols = [col for col in mocap_data_downsample.columns if '_Y' in col]
        z_cols = [col for col in mocap_data_downsample.columns if '_Z' in col]

        left_foot_test_x = mocap_data_downsample.loc[:, ["LANK_X"]]
        left_foot_test_y = mocap_data_downsample.loc[:, ["LANK_Y"]]
        left_foot_test_z = mocap_data_downsample.loc[:, ["LANK_Z"]]

        mocap_data_downsample.loc[:, x_cols] = mocap_data_downsample.loc[:, x_cols].sub(
            left_foot_test_x['LANK_X'], axis=0)
        mocap_data_downsample.loc[:, y_cols] = mocap_data_downsample.loc[:, y_cols].sub(
            left_foot_test_y['LANK_Y'], axis=0)
        mocap_data_downsample.loc[:, z_cols] = mocap_data_downsample.loc[:, z_cols].sub(
            left_foot_test_z['LANK_Z'], axis=0)

        head_x = mocap_data_downsample.loc[:, ["RFHD_X"]]
        head_y = mocap_data_downsample.loc[:, ["RFHD_Y"]]
        head_z = mocap_data_downsample.loc[:, ["RFHD_Z"]]

        people_height = head_z

        mocap_data_downsample.loc[:, x_cols] = mocap_data_downsample.loc[:, x_cols].div(
            people_height['RFHD_Z'], axis=0)
        mocap_data_downsample.loc[:, y_cols] = mocap_data_downsample.loc[:, y_cols].div(
            people_height['RFHD_Z'], axis=0)
        mocap_data_downsample.loc[:, z_cols] = mocap_data_downsample.loc[:, z_cols].div(
            people_height['RFHD_Z'], axis=0)

        ###
        Row_list = []

        # Iterate over each row to turn (34, 3) into (1, 102)
        for index, rows in mocap_data_downsample.iterrows():
            rows = rows.fillna(0)
            my_list = rows.values.tolist()
            my_list_per3 = [my_list[i:i+3] for i in range(0, len(my_list), 3)]
            Row_list.append(flatten(my_list_per3))

        print("Row_list len:", len(Row_list))
        keypoint_list.append(Row_list)
        keypoint_dictionary[this_performer][name_code] = Row_list

        target_dict[name_code] = np.asarray(Row_list)

        if not os.path.exists('./preprocessed_data_save_motion_faster/motion/'):
            os.makedirs('./preprocessed_data_save_motion_faster/motion/')
        motion_data_output = open('./preprocessed_data_save_motion_faster/motion/' +
                                  name_code + '_motion_data.pkl', 'wb')
        pickle.dump(np.asarray(Row_list), motion_data_output)
        motion_data_output.close()
        # load_pickle
        # midi_data_input = open('./preprocessed_data_save/midi/' +
        #                        this_performer + '_' + name_code + '_midi_data.pkl', 'rb')
        # midi_data = pickle.load(midi_data_input)
        # midi_data_input.close()


# mocap_data_downsample = pd.read_pickle('./preprocessed_data_save/motion_data.pkl')
# print("MIDI length:")
# print(piano_roll_length_list)
# print("Motion length:")
# for i in range(0, 110):
#     print(len(keypoint_list[i]))

each_midi_motion_len_compare = {}
for this_performer in performaer_list:
    for piece_name in piece_list:
        each_midi_motion_len_compare[this_performer + '_' + piece_name] = {}
        # each_midi_motion_len_compare[this_performer + '_' + piece_name]["midi"] = len(
        # piano_roll_dictionary[this_performer][this_performer + '_' + piece_name])
        each_midi_motion_len_compare[this_performer + '_' + piece_name]["motion"] = len(
            keypoint_dictionary[this_performer][this_performer + '_' + piece_name])

print("each_midi_motion_len_compare")
print(each_midi_motion_len_compare)
# piano_roll_dictionary[this_performer][this_performer + '_' + piece_list[0]].shape
# keypoint_dictionary[this_performer][this_performer + '_' + piece_list[0]].shape
df_each_midi_motion_len_compare = pd.DataFrame(each_midi_motion_len_compare)
print(df_each_midi_motion_len_compare)
print(df_each_midi_motion_len_compare.T.to_markdown())
