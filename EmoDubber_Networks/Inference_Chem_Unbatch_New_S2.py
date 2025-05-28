import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import argparse
import datetime as dt

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import json
from hifigan.models import Generator as HiFiGAN

from hifigan.config import v1

from hifigan.denoiser import Denoiser

from hifigan.env import AttrDict

from models.EmoDubber import EmoDubber_all

from text_fs import text_to_sequence


from utils.utils import assert_model_downloaded, get_user_data_dir, intersperse

from tqdm import tqdm

import random
import yaml

yaml_path = 'configs/data/Chem_dataset.yaml'
with open(yaml_path, 'r') as file:
    yaml_data = yaml.safe_load(file)
data_stats = yaml_data.get('data_statistics', {})
mel_mean_infer = data_stats.get('mel_mean')
mel_std_infer = data_stats.get('mel_std')

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)


def process_text_FS2(i: int, text: str, device: torch.device):
    x = torch.IntTensor(text_to_sequence(text, ['english_cleaners'])).long().unsqueeze(0).to(device)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return {"x_orig": text, "x": x, "x_lengths": x_lengths}


def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def assert_required_models_available(args):
    model_path = args.checkpoint_path
    vocoder_path = args.vocoder_checkpoint_path
    return {"EmoDubber_Network": model_path, "EmoDubber_16Khz_Vocoder": vocoder_path}


def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def load_hifigan_my16K(checkpoint_path, device):
    with open("Vocoder_16KHz/config.json", "r") as f:
        config_16K = json.load(f)
        
    h = AttrDict(config_16K)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def load_vocoder(vocoder_name, checkpoint_path, device):
    vocoder = None
    
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    elif vocoder_name in ("16K_V0", "16K_V1"): 
        vocoder = load_hifigan_my16K(checkpoint_path, device)
    else:
        raise NotImplementedError(
            f"Vocoder {vocoder_name} not implemented! define a load_<<vocoder_name>> method for it"
        )
    denoiser = Denoiser(vocoder, mode="zeros")
    print(f"[+] {vocoder_name} loaded!")
    return vocoder, denoiser


def load_EmoDubber(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = EmoDubber_all.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()

    print(f"[+] {model_name} loaded!")
    return model



def to_waveform(mel, vocoder, denoiser=None):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()

    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    sf.write(folder / f"{filename}.wav", output["waveform"], 16000, "PCM_24")
    return folder.resolve() / f"{filename}.wav"


def validate_args(args):
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.steps > 0, "Number of ODE steps must be greater than 0"

    if args.checkpoint_path is None:
        args = args
    else:
        if args.speaking_rate is None:
            args.speaking_rate = 1.0

    if args.batched:
        assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"

    return args

   
@torch.inference_mode()
def cli():
    parser = argparse.ArgumentParser(
        description="EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing"
    )
    
    # Path
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default = "Emodubber_chem_1399.ckpt",
        help="Path to the custom model checkpoint",
    )
    

    parser.add_argument(
        "--vocoder_checkpoint_path",
        type=str,
        default = "",
        help="Path to the 16K Vocoder checkpoint",
    )

    parser.add_argument(
        "--Val_list",
        type=str,
        default = "",
        help="Path to the Val Path",
    )
    
    parser.add_argument(
        "--Silent_Lip",
        type=str,
        default = "",
        help="Path to the Lip Feature",
    )
    
    parser.add_argument(
        "--Silent_Face",
        type=str,
        default = "",
        help="Path to the Face Feature",
    )
    
    parser.add_argument(
        "--Refence_audio",
        type=str,
        default = "",
        help="Path to the Refence audio feature",
    )

    parser.add_argument(
        "--Set2_list",
        type=str,
        default = "",
        help="Path to the Setting2 list, Same Speaker from differenct clip as reference audio",
    )
    
    # Fix 
    parser.add_argument(
        "--vocoder",
        type=str,
        default="16K_V0",
        help="Vocoder to use (default: will use the one suggested with the pretrained model))",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="EmoDubber_Model",
        help="Model to use",
    )
    
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=None,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of ODE steps  (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--denoiser_strength",
        type=float,
        default=0.00025,
        help="Strength of the vocoder bias denoiser (default: 0.00025)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument("--batched", action="store_true", help="Batched inference (default: False)")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size only useful when --batched (default: 32)"
    )

    args = parser.parse_args()

    args = validate_args(args)
    device = get_device(args)


    paths = assert_required_models_available(args)

    if args.checkpoint_path is not None:
        paths["EmoDubber_Network"] = args.checkpoint_path
        args.model = "custom_model"

    model = load_EmoDubber(args.model, paths["EmoDubber_Network"], device)
    vocoder, denoiser = load_vocoder(args.vocoder, paths["EmoDubber_16Khz_Vocoder"], device)
    
    
    set_all_random_seed(1234)

    val_path = args.Val_list

    
    data_dict = {}


    with open(args.Set2_list, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            key = parts[0]  
            value = parts[2]  
            data_dict[key] = value
       
       
    with open(
            os.path.join(val_path), "r", encoding="utf-8"
    ) as f:
        for line in tqdm(f.readlines()):
            base_name, base_name_path, phoneme, args.text = line.strip("\n").split("|")
        
            texts = get_texts(args)
            
            reference_audio = data_dict[base_name_path.split('/')[-1].split(".wav")[0]]

            spk_path = os.path.join(
                        args.Refence_audio,
                        "SPK-{}.npy".format(reference_audio),
                    )

            Speaker_GE2E_ID = torch.from_numpy(np.load(spk_path)).unsqueeze(0).to(device)


            lip_embedding_path = os.path.join(
                            args.Silent_Lip,
                            "lipmotion-{}.npy".format(base_name_path.split('/')[-1].split(".wav")[0]),
                        )   
            lip_embedding = torch.from_numpy(np.load(lip_embedding_path)).float().to(device)
            
            lip_embedding_length = torch.tensor(lip_embedding.shape[0], dtype=torch.long)
            lip_embedding_length = lip_embedding_length.unsqueeze(0).to(device)
            
            lip_embedding = lip_embedding.unsqueeze(0)

            VA_path = os.path.join(
                            args.Silent_Face,
                            "{}-feature-{}.npy".format(base_name_path.split('/')[-1].split(".wav")[0][:-4], base_name_path.split('/')[-1].split(".wav")[0]),
                        ) 
            feature_256 = np.load(VA_path)
            feature_256 = torch.from_numpy(feature_256[:lip_embedding.shape[1], :]).unsqueeze(0).float().to(device)

            unbatched_synthesis(args, device, model, vocoder, denoiser, texts, Speaker_GE2E_ID, lip_embedding, lip_embedding_length, base_name_path, base_name, phoneme,feature_256)
    
  



class BatchedSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, processed_texts):
        self.processed_texts = processed_texts

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        return self.processed_texts[idx]


def batched_collate_fn(batch):
    x = []
    x_lengths = []

    for b in batch:
        x.append(b["x"].squeeze(0))
        x_lengths.append(b["x_lengths"])

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x_lengths = torch.concat(x_lengths, dim=0)
    return {"x": x, "x_lengths": x_lengths}




def unbatched_synthesis(args, device, model, vocoder, denoiser, texts, Speaker_GE2E_ID, lip_embedding, lip_embedding_length, base_name_path, base_name, phoneme, feature_256):
    total_rtf = []
    total_rtf_w = []
    for i, text in enumerate(texts):
        i = i + 1
        text_phoneme = phoneme
        text_processed = process_text_FS2(i, text_phoneme, device)
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            lip_embedding, 
            lip_embedding_length,
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=Speaker_GE2E_ID,
            VAfeature = feature_256,
            length_scale=args.speaking_rate,
            data_mel_mean = mel_mean_infer,
            data_mel_std = mel_std_infer,
        )
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 16000 / (output["waveform"].shape[-1])
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)
        location = save_to_folder(base_name, output, os.path.join(args.output_folder, "Setting2_Chem_outputWav_Step_{}".format(args.checkpoint_path.split('/')[-1].split('=')[-1].split('.ckpt')[0])))



def print_config(args):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Temperature: {args.temperature}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Number of ODE steps: {args.steps}")
    print(f"\t- Speaker: {args.spk}")


def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    cli()

