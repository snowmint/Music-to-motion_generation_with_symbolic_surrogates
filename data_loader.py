import os
import random

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from data_utils import read_midi, read_csv


class MidiMotionDataSet(Dataset):
    def __init__(self, file_list_txt):  # 定義路徑，建立 file list # , midi_sr, motion_sr
        self.read_data = {}
        self.read_data["midi"] = []
        self.read_data["motion"] = []
        self.dataset_len = 0
        self.piece_count = 0

        with open(file_list_txt, 'r') as file:
            lines = file.readlines()

        piece_count = 0
        # already done for preprocess
        for line in lines:  # midi_list replace 'midi' with 'motion' then get motion data.

            midi_file_path = "./" + line.rstrip()
            # replace all "midi" with "motion"
            motion_file_path = "./" + line.rstrip().replace("midi", "motion")

            self.read_data["midi"].append(midi_file_path)
            self.read_data["motion"].append(motion_file_path)
            piece_count += 1

        self.dataset_len = piece_count * 100  # 2200
        self.piece_count = piece_count
        print("self.piece_count: ", self.piece_count)
        self.performer_namecodes = self.read_data['midi']
        # print(self.performer_namecodes)
        for performer_piece in self.performer_namecodes:
            self.music_files_index = {
                performer_piece: index for index, file_path in enumerate(self.read_data["midi"])}
        print("dataset_len: ", self.dataset_len)

    def __len__(self):  # len / batch_size = 1 epoch have how many batch
        return self.dataset_len  # index need to fit length

    def __getitem__(self, index):  # 這裡才要讀資料
        # load_pickle with index
        # print("index", index)
        midi_data_input = open(
            self.read_data["midi"][index % self.piece_count], 'rb')
        motion_data_input = open(
            self.read_data["motion"][index % self.piece_count], 'rb')

        midi_data = pickle.load(midi_data_input)
        motion_data = pickle.load(motion_data_input)

        midi_data_input.close()
        motion_data_input.close()

        # TODO: 1. Should concat all data to one array, and then split to sequence_len=512 segment.
        # TODO:     >> If sequence_len cannot divide by 512, then eliminate it.
        # TODO: 2. May use sliding window to split more segment.
        window_size = 512

        left_edge = random.randint(0, len(midi_data) - window_size)
        # print(left_edge)
        segment_midi = np.array(midi_data[left_edge:left_edge+window_size])
        segment_motion = np.array(motion_data[left_edge:left_edge+window_size])
        # print("segment_midi", segment_midi.shape)
        # print("segment_motion", segment_motion.shape)

        # midi_data = midi_data[None, :]
        # motion_data = motion_data[None, :]

        # window_size = 512
        # midi_samples = []
        # motion_samples = []
        # print(midi_data.shape)

        # for midi_sample, motion_sample in zip(midi_data, motion_data):
        #     print("midi_sample", midi_sample.shape)
        #     print("midi_sample.size(0)", midi_sample.shape[0])
        #     midi_sample_len = midi_sample.shape[0]-1
        #     for i in range(0, midi_sample.shape[0], window_size): #or window_size/2
        #         segment_midi = np.array(midi_sample[i:i+window_size])
        #         segment_motion = np.array(motion_sample[i:i+window_size])

        #         segment_midi_len = len(segment_midi)
        #         print("before: ", len(segment_midi))
        #         if segment_midi_len < window_size:
        #             segment_midi = midi_sample[(midi_sample_len-window_size):(midi_sample_len-window_size) + window_size]
        #             segment_motion = motion_sample[(midi_sample_len-window_size):(midi_sample_len-window_size) + window_size]
        #             print("after: ", len(segment_midi))
        #         midi_samples.append(segment_midi)
        #         motion_samples.append(segment_motion)
        #         print("segment_midi.shape:", segment_midi.shape)
        #         print("segment_motion.shape:", segment_motion.shape)

        # print(len(segment_midi), len(segment_motion))
        return segment_midi, segment_motion

        # , self.music_list[index]
        # print(midi_data.shape)
        # print(motion_data.shape)
        # return midi_data, motion_data  # self.performer_namecodes[index]


class MidiMotionValDataSet(Dataset):
    def __init__(self, file_list_txt):  # 定義路徑，建立 file list # , midi_sr, motion_sr
        self.read_data = {}
        self.read_data["midi"] = []
        self.read_data["motion"] = []
        self.dataset_len = 0
        self.max_len = 8952

        with open(file_list_txt, 'r') as file:
            lines = file.readlines()

        piece_count = 0
        # already done for preprocess
        for line in lines:  # midi_list replace 'midi' with 'motion' then get motion data.

            midi_file_path = "./" + line.rstrip()
            # replace all "midi" with "motion"
            motion_file_path = "./" + line.rstrip().replace("midi", "motion")

            self.read_data["midi"].append(midi_file_path)
            self.read_data["motion"].append(motion_file_path)
            piece_count += 1

        self.dataset_len = piece_count  # 5
        self.performer_namecodes = self.read_data['midi']
        # print(self.performer_namecodes)
        for performer_piece in self.performer_namecodes:
            self.music_files_index = {
                performer_piece: index for index, file_path in enumerate(self.read_data["midi"])}
        print("val_dataset_len", self.dataset_len)

    def __len__(self):  # len / batch_size = 1 epoch have how many batch
        return self.dataset_len  # index need to fit length

    def __getitem__(self, index):  # 這裡才要讀資料
        # load_pickle with index
        # print("index", index)
        # | vio01_Wind_S1_T2    |   5281 |
        # | vio02_Wind_S1_T2    |   6061 |
        # | vio03_Wind_S1_T2    |   6069 |
        # | vio04_Wind_S1_T2    |   4525 |
        # | vio05_Wind_S1_T2    |   6706 |
        midi_data_input = open(self.read_data["midi"][index], 'rb')
        motion_data_input = open(self.read_data["motion"][index], 'rb')

        midi_data = pickle.load(midi_data_input)
        motion_data = pickle.load(motion_data_input)
        print("len(midi_data)", len(midi_data))
        print("len(motion_data)", len(motion_data))

        midi_data_input.close()
        motion_data_input.close()

        pad_len = self.max_len - len(midi_data)

        # F.pad(midi_data, (0,0,0,pad_len), value = 0)
        midi_data_pad = np.pad(midi_data, pad_width=(
            (0, pad_len), (0, 0)), constant_values=0)
        motion_data_pad = np.pad(motion_data, pad_width=(
            (0, pad_len), (0, 0)), constant_values=0)

        return midi_data_pad, motion_data_pad  # Full data for evaluation

# class MidiMotionCollate():
#     def __init__(self, device):
#         super().__init__()
#         self.device = device

#     def __call__(self, batch):
#         B = len(batch)

#         # midi_dim = batch[0][0][0].shape[1]  # 128
#         # motion_dim = batch[0][1][0].shape[1]  # 102
#         # print("midi_dim: ", midi_dim)
#         # print("motion_dim: ", motion_dim)
#         # # have same length
#         # midi_len = [data[0][0].shape[0] for data in batch]
#         # motion_len = [data[0][1].shape[0] for data in batch]

#         # max_len = max(midi_len)
#         # print("max_len:", max_len)

#         pad_midi = torch.zeros((B, max_len, midi_dim), dtype=torch.int32)
#         pad_motion = torch.zeros(
#             (B, max_len, motion_dim), dtype=torch.float)

#         # # pad_mask_midi = torch.arange(midi_len)
#         # # pad_mask_motion = torch.arange(motion_len)

#         for i, (midi_data, motion_data) in enumerate(batch):
#             pad_midi[j, :midi_data[0].shape[0], :] = torch.Tensor(midi_data[0])
#             pad_motion[j, :motion_data[0].shape[0], :] = torch.Tensor(motion_data[0])

#         # torch.Tensor(midi_data).unsqueeze(0)
#         # torch.Tensor(motion_data).unsqueeze(0)
#         print(pad_midi.shape)
#         print(pad_motion.shape)

#         pad_midi = pad_midi.to(self.device)
#         pad_motion = pad_motion.to(self.device)
#         # pad_mask_midi = pad_mask_midi.to(self.device)
#         # pad_mask_motion = pad_mask_motion.to(self.device)

#         return pad_midi, pad_motion
        # return pad_midi, pad_motion, stop_token, pad_mask_midi, pad_mask_motion


class AudioMotionDataSet(Dataset):
    def __init__(self, file_list_txt):  # 定義路徑，建立 file list # , midi_sr, motion_sr
        self.read_data = {}
        self.read_data["audio"] = []
        self.read_data["motion"] = []
        self.dataset_len = 0
        self.piece_count = 0

        with open(file_list_txt, 'r') as file:
            lines = file.readlines()

        piece_count = 0
        # already done for preprocess
        for line in lines:  # audio_list replace 'audio' with 'motion' then get motion data.

            audio_file_path = "./" + line.rstrip()
            # replace all "audio" with "motion"
            motion_file_path = "./" + line.rstrip().replace("audio", "motion")
            # print("motion_file_path", motion_file_path)
            self.read_data["audio"].append(audio_file_path)
            self.read_data["motion"].append(motion_file_path)
            piece_count += 1

        self.dataset_len = piece_count * 100  # 2200
        self.piece_count = piece_count
        print("self.piece_count: ", self.piece_count)
        self.performer_namecodes = self.read_data['audio']
        # print(self.performer_namecodes)
        for performer_piece in self.performer_namecodes:
            self.music_files_index = {
                performer_piece: index for index, file_path in enumerate(self.read_data["audio"])}
        print("dataset_len: ", self.dataset_len)

    def __len__(self):  # len / batch_size = 1 epoch have how many batch
        return self.dataset_len  # index need to fit length

    def __getitem__(self, index):  # 這裡才要讀資料
        # load_pickle with index
        # print("index", index)
        audio_data_input = open(
            self.read_data["audio"][index % self.piece_count], 'rb')
        motion_data_input = open(
            self.read_data["motion"][index % self.piece_count], 'rb')

        audio_data = pickle.load(audio_data_input)
        motion_data = pickle.load(motion_data_input)

        audio_data_input.close()
        motion_data_input.close()

        # TODO: 1. Should concat all data to one array, and then split to sequence_len=512 segment.
        # TODO:     >> If sequence_len cannot divide by 512, then eliminate it.
        # TODO: 2. May use sliding window to split more segment.
        window_size = 512

        left_edge = random.randint(0, len(audio_data) - window_size)
        # print(left_edge)
        segment_audio = np.array(audio_data[left_edge:left_edge+window_size])
        segment_motion = np.array(motion_data[left_edge:left_edge+window_size])
        # print("segment_audio", segment_audio.shape)
        # print("segment_motion", segment_motion.shape)

        return segment_audio, segment_motion


class AudioMotionValDataSet(Dataset):
    def __init__(self, file_list_txt):  # 定義路徑，建立 file list # , audio_sr, motion_sr
        self.read_data = {}
        self.read_data["audio"] = []
        self.read_data["motion"] = []
        self.dataset_len = 0
        self.max_len = 6706

        with open(file_list_txt, 'r') as file:
            lines = file.readlines()

        piece_count = 0
        # already done for preprocess
        for line in lines:  # audio_list replace 'audio' with 'motion' then get motion data.

            audio_file_path = "./" + line.rstrip()
            # replace all "audio" with "motion"
            motion_file_path = "./" + line.rstrip().replace("audio", "motion")

            self.read_data["audio"].append(audio_file_path)
            self.read_data["motion"].append(motion_file_path)
            piece_count += 1

        self.dataset_len = piece_count  # 5
        self.performer_namecodes = self.read_data['audio']
        # print(self.performer_namecodes)
        for performer_piece in self.performer_namecodes:
            self.music_files_index = {
                performer_piece: index for index, file_path in enumerate(self.read_data["audio"])}
        print("val_dataset_len", self.dataset_len)

    def __len__(self):  # len / batch_size = 1 epoch have how many batch
        return self.dataset_len  # index need to fit length

    def __getitem__(self, index):  # 這裡才要讀資料
        # load_pickle with index
        # print("index", index)
        # | vio01_Wind_S1_T2    |   5281 |
        # | vio02_Wind_S1_T2    |   6061 |
        # | vio03_Wind_S1_T2    |   6069 |
        # | vio04_Wind_S1_T2    |   4525 |
        # | vio05_Wind_S1_T2    |   6706 |
        audio_data_input = open(self.read_data["audio"][index], 'rb')
        motion_data_input = open(self.read_data["motion"][index], 'rb')

        audio_data = pickle.load(audio_data_input)
        motion_data = pickle.load(motion_data_input)
        print("len(audio_data)", len(audio_data))
        print("len(motion_data)", len(motion_data))

        audio_data_input.close()
        motion_data_input.close()

        pad_len = self.max_len - len(audio_data)

        # F.pad(midi_data, (0,0,0,pad_len), value = 0)
        audio_data_pad = np.pad(audio_data, pad_width=(
            (0, pad_len), (0, 0)), constant_values=0)
        motion_data_pad = np.pad(motion_data, pad_width=(
            (0, pad_len), (0, 0)), constant_values=0)

        return audio_data_pad, motion_data_pad  # Full data for evaluation

# ===


class AllDataSet(Dataset):
    def __init__(self, file_list_txt):  # 定義路徑，建立 file list # , midi_sr, motion_sr
        self.read_data = {}
        self.read_data["all"] = []
        self.read_data["motion"] = []
        self.dataset_len = 0
        self.piece_count = 0

        with open(file_list_txt, 'r') as file:
            lines = file.readlines()

        piece_count = 0
        # already done for preprocess
        for line in lines:  # audio_list replace 'audio' with 'motion' then get motion data.

            all_file_path = "./" + line.rstrip()
            # replace all "all" with "motion"
            motion_file_path = "./" + line.rstrip().replace("all", "motion")
            # print("motion_file_path", motion_file_path)
            self.read_data["all"].append(all_file_path)
            self.read_data["motion"].append(motion_file_path)
            piece_count += 1

        self.dataset_len = piece_count * 100  # 2200
        self.piece_count = piece_count
        print("self.piece_count: ", self.piece_count)
        self.performer_namecodes = self.read_data['all']
        # print(self.performer_namecodes)
        for performer_piece in self.performer_namecodes:
            self.music_files_index = {
                performer_piece: index for index, file_path in enumerate(self.read_data["all"])}
        print("dataset_len: ", self.dataset_len)

    def __len__(self):  # len / batch_size = 1 epoch have how many batch
        return self.dataset_len  # index need to fit length

    def __getitem__(self, index):  # 這裡才要讀資料
        # load_pickle with index
        # print("index", index)
        all_data_input = open(
            self.read_data["all"][index % self.piece_count], 'rb')
        motion_data_input = open(
            self.read_data["motion"][index % self.piece_count], 'rb')

        all_data = pickle.load(all_data_input)
        motion_data = pickle.load(motion_data_input)

        all_data_input.close()
        motion_data_input.close()

        # TODO: 1. Should concat all data to one array, and then split to sequence_len=512 segment.
        # TODO:     >> If sequence_len cannot divide by 512, then eliminate it.
        # TODO: 2. May use sliding window to split more segment.
        window_size = 512

        left_edge = random.randint(0, len(all_data) - window_size)
        # print(left_edge)
        segment_all = np.array(all_data[left_edge:left_edge+window_size])
        segment_motion = np.array(motion_data[left_edge:left_edge+window_size])
        # print("segment_audio", segment_audio.shape)
        # print("segment_motion", segment_motion.shape)

        return segment_all, segment_motion


class AllValDataSet(Dataset):
    def __init__(self, file_list_txt):  # 定義路徑，建立 file list # , midi_sr, motion_sr
        self.read_data = {}
        self.read_data["all"] = []
        self.read_data["motion"] = []
        self.dataset_len = 0
        self.max_len = 6706

        with open(file_list_txt, 'r') as file:
            lines = file.readlines()

        piece_count = 0
        # already done for preprocess
        for line in lines:  # all_list replace 'all' with 'motion' then get motion data.

            all_file_path = "./" + line.rstrip()
            # replace all "all" with "motion"
            motion_file_path = "./" + line.rstrip().replace("all", "motion")

            self.read_data["all"].append(all_file_path)
            self.read_data["motion"].append(motion_file_path)
            piece_count += 1

        self.dataset_len = piece_count  # 5
        self.performer_namecodes = self.read_data['all']
        # print(self.performer_namecodes)
        for performer_piece in self.performer_namecodes:
            self.music_files_index = {
                performer_piece: index for index, file_path in enumerate(self.read_data["all"])}
        print("val_dataset_len", self.dataset_len)

    def __len__(self):  # len / batch_size = 1 epoch have how many batch
        return self.dataset_len  # index need to fit length

    def __getitem__(self, index):  # 這裡才要讀資料
        # load_pickle with index
        # print("index", index)
        # | vio01_Wind_S1_T2    |   5281 |
        # | vio02_Wind_S1_T2    |   6061 |
        # | vio03_Wind_S1_T2    |   6069 |
        # | vio04_Wind_S1_T2    |   4525 |
        # | vio05_Wind_S1_T2    |   6706 |
        all_data_input = open(self.read_data["all"][index], 'rb')
        motion_data_input = open(self.read_data["motion"][index], 'rb')

        all_data = pickle.load(all_data_input)
        motion_data = pickle.load(motion_data_input)
        print("len(all_data)", len(all_data))
        print("len(motion_data)", len(motion_data))

        all_data_input.close()
        motion_data_input.close()

        pad_len = self.max_len - len(all_data)

        # F.pad(midi_data, (0,0,0,pad_len), value = 0)
        all_data_pad = np.pad(all_data, pad_width=(
            (0, pad_len), (0, 0)), constant_values=0)
        motion_data_pad = np.pad(motion_data, pad_width=(
            (0, pad_len), (0, 0)), constant_values=0)

        return all_data_pad, motion_data_pad  # Full data for evaluation


def get_dataloader(dataset_path, batch_size=1, device='cuda'):
    dataset = MidiMotionDataSet(dataset_path)
    # music_list = dataset.get_music_list()
    # collate_feature = MidiMotionCollate(device)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             shuffle=True,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)  # collate_fn=collate_feature
    return data_loader  # , music_list


def get_audio_dataloader(dataset_path, batch_size=1, device='cuda'):
    dataset = AudioMotionDataSet(dataset_path)
    # music_list = dataset.get_music_list()
    # collate_feature = MidiMotionCollate(device)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             shuffle=True,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)  # collate_fn=collate_feature
    return data_loader  # , music_list


def get_all_dataloader(dataset_path, batch_size=1, device='cuda'):
    dataset = AllDataSet(dataset_path)
    # music_list = dataset.get_music_list()
    # collate_feature = MidiMotionCollate(device)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             shuffle=True,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)  # collate_fn=collate_feature
    return data_loader  # , music_list


def get_val_dataloader(dataset_path, batch_size=1, device='cuda'):
    dataset = MidiMotionValDataSet(dataset_path)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             shuffle=True,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)
    return data_loader


def get_audio_val_dataloader(dataset_path, batch_size=1, device='cuda'):
    dataset = AudioMotionValDataSet(dataset_path)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             shuffle=True,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)
    return data_loader


def get_all_val_dataloader(dataset_path, batch_size=1, device='cuda'):
    dataset = AllValDataSet(dataset_path)
    data_loader = DataLoader(dataset,
                             num_workers=0,
                             shuffle=True,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=False,
                             drop_last=False)
    return data_loader


if __name__ == "__main__":
    dataset_name_path = f"./midi_list.txt"
    dataloader = get_dataloader(dataset_name_path, batch_size=20)

    for pad_midi, pad_motion in dataloader:
        print(pad_midi.shape, pad_motion.shape)
