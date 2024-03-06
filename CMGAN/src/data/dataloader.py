import torch.utils.data
import torchaudio
import os
from utils import *
import random
from natsort import natsorted
from itertools import groupby
from operator import itemgetter

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data.distributed import DistributedSampler


class LibrimixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2, overfit=False):
        self.overfit = False
        self.cut_len = cut_len
        self.mix_dir = os.path.join(data_dir, "mix")
        self.target_dir = os.path.join(data_dir, "target")
        self.target_wav_name = os.listdir(self.target_dir)
        self.target_wav_name = natsorted(self.target_wav_name)
        speaker_mapping = [[v.split("-")[0], v] for v in self.target_wav_name]

        self.speaker_mapping = {}
        for sublist in speaker_mapping:
            if sublist[0] in self.speaker_mapping:
                self.speaker_mapping[sublist[0]].append(sublist[1])
            else:
                self.speaker_mapping[sublist[0]] = [sublist[1]]

        self.resampler = torchaudio.transforms.Resample(8000, 16000)

    def __len__(self):
        return len(self.target_wav_name)

    def __getitem__(self, idx):
        target_file = os.path.join(self.target_dir, self.target_wav_name[idx])

        target_speaker = self.target_wav_name[idx].split("-")[0]
        reference_files = self.speaker_mapping[target_speaker]

        if self.overfit:
            reference_file = reference_files[0]
            for file in reference_files:
                if file != target_file and len(reference_files) > 1:
                    reference_file = file

            reference_file = reference_files[
                random.randint(0, len(reference_files) - 1)
            ]
        else:
            reference_file = reference_files[
                random.randint(0, len(reference_files) - 1)
            ]
        reference_file = os.path.join(self.target_dir, reference_file)

        mix_file = os.path.join(self.mix_dir, self.target_wav_name[idx])

        target_ds, _ = torchaudio.load(target_file)
        mix_ds, _ = torchaudio.load(mix_file)
        reference_ds, _ = torchaudio.load(reference_file)
        reference_ds = self.resampler(reference_ds)

        target_ds = target_ds.squeeze()
        mix_ds = mix_ds.squeeze()
        reference_ds = reference_ds.squeeze()
        length = len(target_ds)
        assert length == len(mix_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            target_ds_final = []
            mix_ds_final = []
            reference_ds_final = []
            for i in range(units):
                target_ds_final.append(target_ds)
                mix_ds_final.append(mix_ds)
                reference_ds_final.append(reference_ds)
            target_ds_final.append(target_ds[: self.cut_len % length])
            mix_ds_final.append(mix_ds[: self.cut_len % length])
            reference_ds_final.append(reference_ds[: self.cut_len % length])
            target_ds = torch.cat(target_ds_final, dim=-1)
            mix_ds = torch.cat(mix_ds_final, dim=-1)
            reference_ds = torch.cat(reference_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            if self.overfit:
                wav_start = 0
            else:
                wav_start = random.randint(0, length - self.cut_len)
            mix_ds = mix_ds[wav_start : wav_start + self.cut_len]
            target_ds = target_ds[wav_start : wav_start + self.cut_len]

            if self.overfit:
                wav_start = 0
            else:
                wav_start = random.randint(0, reference_ds.size(0) - self.cut_len * 2)
            reference_ds = reference_ds[wav_start : wav_start + self.cut_len * 2]

        return target_ds, mix_ds, length, reference_ds


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]

        return clean_ds, noisy_ds, length


def load_data(ds_dir, batch_size, n_cpu, cut_len, overfit=False):
    # torchaudio.set_audio_backend("sox_io")  # in linux
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = LibrimixDataset(train_dir, cut_len, overfit=False)
    test_ds = LibrimixDataset(test_dir, cut_len, overfit=False)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        # sampler=DistributedSampler(train_ds),
        drop_last=True,
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        # sampler=DistributedSampler(test_ds),
        drop_last=False,
        num_workers=n_cpu,
    )

    return train_dataset, test_dataset
