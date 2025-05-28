import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader


from text_fs import text_to_sequence

from utils.audio import mel_spectrogram
from utils.model import fix_len_compatibility, normalize
from utils.utils import intersperse

import os

import ast

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(
        self,
        name,
        lip_embedding_path,
        Speaker_GE2E_ID_path,
        GT_SIM_path,
        pitch_path,
        energy_path,
        VA_path,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None): 
        self.trainset = TextMelDataset(  
            self.hparams.name,
            self.hparams.lip_embedding_path,
            self.hparams.Speaker_GE2E_ID_path,
            self.hparams.GT_SIM_path,
            self.hparams.pitch_path,
            self.hparams.energy_path,
            self.hparams.VA_path,
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        self.validset = TextMelDataset( 
            self.hparams.name,
            self.hparams.lip_embedding_path,
            self.hparams.Speaker_GE2E_ID_path,
            self.hparams.GT_SIM_path,
            self.hparams.pitch_path,
            self.hparams.energy_path,
            self.hparams.VA_path,
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name, # used to choose dataset
        lip_embedding_path,
        Speaker_GE2E_ID_path,
        GT_SIM_path,
        pitch_path,
        energy_path,
        VA_path,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=640, # 1024,
        n_mels=80,
        sample_rate=16000,
        hop_length=160,
        win_length=640,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.datasetname = name

        self.input_lip_embedding_path = lip_embedding_path
        self.input_Speaker_GE2E_ID_path = Speaker_GE2E_ID_path
        self.input_GT_SIM_path = GT_SIM_path
        self.input_pitch_path = pitch_path
        self.input_energy_path =  energy_path
        self.input_VA_path = VA_path


        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank # add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
                    
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)


    def extract_window(self, mouthroi, mel, info, filepath):
        
        self.audio_stft_hop = 160
        self.videos_window_size = 25
        
        
        hop =  self.audio_stft_hop

        # vid : T,C,H,W
        vid_2_aud = info['audio_fps'] / info['video_fps'] / hop # 4.0

        mel_window_size = int(mouthroi.shape[0] * vid_2_aud)
        
        if mel.shape[1] >= mel_window_size:
            mel_ = mel[:, : mel_window_size]
        else:
            video_window_size = int(mel.shape[-1] / vid_2_aud)
            
            mel_window_size_re = int(video_window_size * vid_2_aud)
            
            mel_ = mel[:, : mel_window_size_re]
            mouthroi = mouthroi[: video_window_size, :]
            
        
        return mouthroi, mel_
    
    def get_datapoint(self, filepath_and_text):
        
        if self.datasetname == "Chem":
            # 
            if self.n_spks > 1:
                filepath, spk, text = (
                    filepath_and_text[0],
                    int(filepath_and_text[1]),
                    filepath_and_text[2],
                )
            else:
                basicname, filepath, cleaned_phoneme, cleaned_text = filepath_and_text[0], filepath_and_text[1], filepath_and_text[2], filepath_and_text[3] # , filepath_and_text[2]
                spk = None
            
            
            text = self.get_text(cleaned_phoneme, add_blank=self.add_blank)

            mel = self.get_mel(filepath)        
        
            lip_embedding_path = os.path.join(
                            self.input_lip_embedding_path,
                            "lipmotion-{}.npy".format(filepath.split('/')[-1].split(".wav")[0]),
                        )        
            lip_embedding = torch.from_numpy(np.load(lip_embedding_path)).float()
            

            Speaker_GE2E_ID_path = os.path.join(
                            self.input_Speaker_GE2E_ID_path,
                            "SPK-{}.npy".format(filepath.split('/')[-1].split(".wav")[0]),
                        )
            Speaker_GE2E_ID = torch.from_numpy(np.load(Speaker_GE2E_ID_path))

            durations = self.get_durations(filepath, text) if self.load_durations else None
            
            
            GT_SIM_path = os.path.join(
                            self.input_GT_SIM_path,
                            "16KChem_GTSIM-{}.npy".format(filepath.split('/')[-1].split(".wav")[0]),
                        )        
            GT_SIM = torch.from_numpy(np.load(GT_SIM_path)).float()
            
            pitch_path = os.path.join(
                            self.input_pitch_path,
                            "Pitch-{}.npy".format(filepath.split('/')[-1].split(".wav")[0]),
                        ) 
            pitch = np.load(pitch_path)
            pitch = pitch[:len(text)]
            pitch = torch.from_numpy(pitch).float()

            energy_path = os.path.join(
                            self.input_energy_path,
                            "Energy-{}.npy".format(filepath.split('/')[-1].split(".wav")[0]),
                        ) 
            energy = np.load(energy_path)
            energy = energy[:len(text)]
            energy = torch.from_numpy(energy).float()
            
            VA_path = os.path.join(
                            self.input_VA_path,
                            "{}-feature-{}.npy".format(filepath.split('/')[-1].split(".wav")[0][:-4], filepath.split('/')[-1].split(".wav")[0]),
                        ) 
            feature_256 = np.load(VA_path)
            feature_256 = torch.from_numpy(feature_256[:lip_embedding.shape[0], :]).float()
            
            
            
            
            text = text[:GT_SIM.shape[0]]
            
            
            mel = mel[:, : lip_embedding.shape[0]*4]
        
        if self.datasetname == "GRID":
            if self.n_spks > 1:
                filepath, spk, text = (
                    filepath_and_text[0],
                    int(filepath_and_text[1]),
                    filepath_and_text[2],
                )
            else:
                filepath, cleaned_text, cleaned_phoneme = filepath_and_text[0], filepath_and_text[1], filepath_and_text[2] # , filepath_and_text[3] # , filepath_and_text[2]
                spk = None
            
            
            text = self.get_text(cleaned_phoneme, add_blank=self.add_blank)
            
            mel = self.get_mel(filepath)


            c_n = filepath.split('/')[-1].split('.wav')[0]
            base_n = filepath.split('/')[-2]
        
        
            lip_embedding_path = os.path.join(
                            self.input_lip_embedding_path,
                            "{}-face-{}.npy".format(base_n, c_n),
                        )      
            lip_embedding = torch.from_numpy(np.load(lip_embedding_path)).float()
            
            info = {'audio_fps': 16000, 'video_fps': 25.0} 

            lip_embedding, mel = self.extract_window(lip_embedding, mel, info, filepath)


            Speaker_GE2E_ID_path = os.path.join(
                            self.input_Speaker_GE2E_ID_path, base_n, c_n+'.npy'
                        )   
            Speaker_GE2E_ID = torch.from_numpy(np.load(Speaker_GE2E_ID_path))


            durations = self.get_durations(filepath, text) if self.load_durations else None
            
            
            GT_SIM_path = os.path.join(
                            self.input_GT_SIM_path,
                            "16KGRID_GTSIM-{}.npy".format(c_n),
                        )        
            GT_SIM = torch.from_numpy(np.load(GT_SIM_path)).float()
            
            pitch_path = os.path.join(
                            self.input_pitch_path,
                            "Pitch-{}.npy".format(c_n),
                        ) 
            pitch = np.load(pitch_path)
            pitch = torch.from_numpy(pitch).float()

            energy_path = os.path.join(
                            self.input_energy_path,
                            "Energy-{}.npy".format(c_n),
                        ) 
            energy = np.load(energy_path)
            energy = torch.from_numpy(energy).float()
            
            VA_path = os.path.join(
                            self.input_VA_path,
                            "{}-feature-{}.npy".format(base_n, c_n),
                        ) 
            feature_256 = np.load(VA_path)
            feature_256 = torch.from_numpy(feature_256[:lip_embedding.shape[0], :]).float()
            
            
            
        

        return {"x": text, "y": mel, "spk": Speaker_GE2E_ID, "filepath": filepath, "x_text": cleaned_text, "durations": durations, "lip_embedding": lip_embedding, "GT_SIM": GT_SIM, "pitch": pitch, "energy": energy, "feature_256": feature_256}


    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel




    def get_text(self, phoneme, add_blank=True):
        
        phone = text_to_sequence(phoneme, ['english_cleaners'])
        
        text_norm = torch.IntTensor(phone).long()
        return text_norm
    

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        
        
        
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats = batch[0]["y"].shape[-2]
        
        E_lip_embedding = batch[0]["lip_embedding"].shape[-1]
        
        lip_max_length = max([item["lip_embedding"].shape[-2] for item in batch])

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        
        pitch = torch.zeros((B, x_max_length), dtype=torch.float32)
        energy = torch.zeros((B, x_max_length), dtype=torch.float32)
        
        GT_sim = torch.zeros((B, x_max_length, lip_max_length), dtype=torch.long)

        lip = torch.zeros((B, lip_max_length, E_lip_embedding), dtype=torch.float32)
        
        VAfeature = torch.zeros((B, lip_max_length, 256), dtype=torch.float32)
        
        
        
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths, lip_lengths = [], [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_, lip_, GT_sim_, feature_256_, pitch_, energy_ = item["y"], item["x"], item["lip_embedding"], item["GT_SIM"], item["feature_256"], item["pitch"], item["energy"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            lip_lengths.append(lip_.shape[-2])
            
            y[i, :, : y_.shape[-1]] = y_

            x[i, : x_.shape[-1]] = x_
            lip[i, : lip_.shape[-2], :] = lip_
            
            VAfeature[i, : lip_.shape[-2], :] = feature_256_
            
            
            
            
            pitch[i, : x_.shape[-1]] = pitch_
            energy[i, : x_.shape[-1]] = energy_
            
            GT_sim[i, : x_.shape[-1], :lip_.shape[-2]] = GT_sim_
            
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        lip_lengths = torch.tensor(lip_lengths, dtype=torch.long)
        
        spks = torch.stack(spks).float()


        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "lip": lip,
            "lip_lengths": lip_lengths,
            "GT_sim": GT_sim,
            "VAfeature": VAfeature,
            "pitch": pitch,
            "energy": energy,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }

