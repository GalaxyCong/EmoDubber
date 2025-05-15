import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

import torch
import librosa
import argparse
import numpy as np
from tqdm import tqdm
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
# predictor = torch.hub.load("/data/conggaoxiang/.cache/torch/hub/tarepan-SpeechMOS-v1.2.0", "utmos22_strong", trust_repo=True,source='local')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor= predictor.to(device)

def predict_mos(wav_path):

    wave, sr = librosa.load(wav_path, sr=None, mono=True)

    score = predictor(torch.from_numpy(wave).unsqueeze(0).to(device), sr).detach().cpu().numpy()
    
    return score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--pred_path",
        type=str,
        required=True,
        help="path to pred wav files",
    )
    args = parser.parse_args()

    wav_list = os.listdir(args.pred_path)
    mos_score_list = []
    for i in tqdm(range(len(wav_list))):
        mos_score_list.append(predict_mos(os.path.join(args.pred_path, wav_list[i])))

    mos_score_list = np.array(mos_score_list)
    print("The mean predict mos of {} is {:.4f}".format(args.pred_path, mos_score_list.mean()))


