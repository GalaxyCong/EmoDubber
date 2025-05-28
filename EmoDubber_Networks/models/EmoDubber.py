import datetime as dt
import math
# import random
import torch
import torch.nn.functional as F
# import sys
import os
import torch.nn as nn
from monotonic_align import mask_from_lens, maximum_path
import numpy as np
import json

from utils.tools import get_mask_from_lengths, pad
import utils
from models.baselightningmodule import BaseLightningClass
from models.components.flow_matching import CFM
from models.components.text_encoder import TextEncoder
from utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)


from nnet import layers
from nnet import blocks
from nnet import attentions
from nnet import normalizations



from transformer import PostNet, Decoder_Condition
from MMAttention.sma import StepwiseMonotonicMultiheadAttention
import utils.monotonic_align as monotonic_align
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


log = utils.get_pylogger(__name__)


class EmoDubber_all(BaseLightningClass):
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        use_precomputed_durations=False,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        self.n_vocab = n_vocab
        self.CTC_classifier_mel = CTC_classifier_mel(self.n_vocab) 
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        self.audio_visual_encoder = ConformerInterCTC(
            dim_model=256,
            num_blocks=5,
            interctc_blocks=[2],
            vocab_size=self.n_vocab,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": 4, "attn_drop_rate": 0.0, "num_pos_embeddings": 100000, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": 15}},
            ff_ratio=4,
            drop_rate=0.1,
            pos_embedding=None, 
            mask=attentions.Mask(), 
            conv_stride=2, 
            batch_norm=True,
            loss_prefix="f_ctc"
        )
        
        self.variance_adaptor_PE = VarianceAdaptor(encoder.VarianceAdaptor.NameforStats)
        
        self.lip_encoder = torch.nn.Conv1d(512, 256, 1) 
        
        self.decoder_Condition = Decoder_Condition(encoder)
        
        self.CTC_criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum').cuda()
        
        weight_init = "default"
        bias_init = "default"
        act_fun = "Swish"
        self.layers_fused = nn.Sequential(
            Linear(512, 1024, weight_init=weight_init, bias_init=bias_init),
            act_dict[act_fun](),
            Linear(1024, 256, weight_init=weight_init, bias_init=bias_init),
        )

        self.mel_linear = nn.Linear(
            256,
            80,)
        
        self.postnet = PostNet()
        
        self.attn_lip_text  = StepwiseMonotonicMultiheadAttention(256, 256//4, 256//4)

        self.upsample_conv2d = torch.nn.ModuleList()
        for s in encoder.mel_upsample:
            conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
            torch.nn.init.kaiming_normal_(conv_trans2d.weight)
            self.upsample_conv2d.append(conv_trans2d)
        
        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        self.update_data_statistics(data_statistics)

    def get_pittch_loss(self, pitch_predictions, pitch_targets, masks):
        masks = ~masks
        pitch_targets.requires_grad = False
        pitch_predictions = pitch_predictions.masked_select(masks)
        pitch_targets = pitch_targets.masked_select(masks)
        pitch_loss = F.l1_loss(pitch_predictions, pitch_targets)
        
        return pitch_loss

    def get_energy_loss(self, energy_predictions, energy_targets, masks):
        masks = ~masks
        energy_targets.requires_grad = False
        energy_predictions = energy_predictions.masked_select(masks)
        energy_targets = energy_targets.masked_select(masks)
        
        energy_loss = F.l1_loss(energy_predictions, energy_targets)
        
        return energy_loss
    
    @torch.inference_mode()
    def synthesise(self, x, x_lengths, Lip, lip_lengths, n_timesteps, temperature=1.0, spks=None, GT_sim=None, VAfeature=None, pitch=None, energy=None, mel_LLM=None, length_scale=1.0, data_mel_mean=0.0, data_mel_std=1.0):
        t = dt.datetime.now() # For RTF computation


        y_max_length = int(4.0*Lip.shape[1])
        y_lengths = 4.0 * lip_lengths
        
        mu_x, x_mask = self.encoder(x, x_lengths, spks)
        
        lip_masks = get_mask_from_lengths(lip_lengths, Lip.shape[1])

        lip_embedding = self.lip_encoder(Lip.transpose(2,1)).transpose(2,1)
        
        if x_mask is not None:
            src_masks = (x_mask == 0).squeeze(1)
            slf_attn_mask_text = src_masks.unsqueeze(1).expand(-1, lip_masks.size(1), -1)
            slf_attn_mask_lip = lip_masks.unsqueeze(1).expand(-1, src_masks.size(1), -1)
            slf_attn_mask = slf_attn_mask_lip.transpose(1,2) | slf_attn_mask_text

        (
            mu_x_p,
            _,
            _,
        ) = self.variance_adaptor_PE(
            mu_x,
            VAfeature,
            lip_masks,
            src_masks,
            e_control = 1.0,
        )
        
        output_text_lip, AV_attn, _ =self.attn_lip_text(lip_embedding, mu_x_p, mu_x_p, x_lengths, mask=slf_attn_mask, query_mask=lip_masks.unsqueeze(2))

        mask_sim = mask_from_lens(AV_attn.transpose(1,2), x_lengths, lip_lengths)
        alignment = maximum_path(AV_attn.transpose(1,2).contiguous(), mask_sim)

        pred_txt_un_phoneme = torch.matmul(alignment.transpose(1, 2), mu_x.transpose(1,2))

        x_fused = torch.cat([pred_txt_un_phoneme, output_text_lip], dim=-1)
        x_fused = self.layers_fused(x_fused)
        x_fused, _ = self.audio_visual_encoder(x_fused, lip_lengths)
        
        x_fused = torch.unsqueeze(x_fused.transpose(1,2), dim=1)
        x_fused = F.leaky_relu(self.upsample_conv2d[0](x_fused), 0.4)
        x_fused = F.leaky_relu(self.upsample_conv2d[1](x_fused), 0.4)
        x_fused = torch.squeeze(x_fused, dim=1).transpose(1,2)
        
        mel_masks_formerDecoder = get_mask_from_lengths(y_lengths, y_max_length)
        pred_d_mu, _ = self.decoder_Condition(x_fused, mel_masks_formerDecoder, spks)
        
        pred_d_mu = self.mel_linear(pred_d_mu)
        postnet_output = self.postnet(pred_d_mu, spks) + pred_d_mu
        
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(torch.float32)
        
        mu_y = postnet_output.transpose(1,2) * y_mask

        encoder_outputs = mu_y[:, :, :y_max_length]
        
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 16000 / (decoder_outputs.shape[-1] * 160)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": AV_attn[:, :, :],
            # "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel": denormalize(decoder_outputs, data_mel_mean, data_mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }


    def forward(self, x, x_lengths, y, y_lengths, Lip, lip_lengths, spks=None, GT_sim=None, VAfeature=None, pitch=None, energy=None, mel_LLM=None, out_size=None, cond=None, durations=None):

        mu_x, x_mask = self.encoder(x, x_lengths, spks)
        
        # predic energy and pitch
        lip_masks = get_mask_from_lengths(lip_lengths, Lip.shape[1])
        lip_embedding = self.lip_encoder(Lip.transpose(2,1)).transpose(2,1)
        
        if x_mask is not None:
            src_masks = (x_mask == 0).squeeze(1)
            slf_attn_mask_text = src_masks.unsqueeze(1).expand(-1, lip_masks.size(1), -1)
            slf_attn_mask_lip = lip_masks.unsqueeze(1).expand(-1, src_masks.size(1), -1)
            slf_attn_mask = slf_attn_mask_lip.transpose(1,2) | slf_attn_mask_text

        
        (
            mu_x_p,
            e_predictions,
            p_predictions,
        ) = self.variance_adaptor_PE(
            mu_x,
            VAfeature,
            lip_masks,
            src_masks,
            energy_target = energy,
            pitch_target = pitch,
            e_control = 1.0,
        )
        energy_loss = 1.2*self.get_energy_loss(e_predictions, energy, src_masks)
        pitch_loss_v = 1.2*self.get_pittch_loss(p_predictions, pitch, src_masks)
        
        
        output_text_lip, AV_attn, _ =self.attn_lip_text(lip_embedding, mu_x_p, mu_x_p, x_lengths, mask=slf_attn_mask, query_mask=lip_masks.unsqueeze(2))
        

        if GT_sim is not None:
            similarity = AV_attn.transpose(1,2)
            if src_masks is not None:
                similarity = similarity * (1 - src_masks.float())[:, :, None]
            if lip_masks is not None:
                similarity = similarity * (1 - lip_masks.float())[:, None, :]
            sim_exp = torch.exp(similarity/0.1)
            mu_a = torch.sum(GT_sim*sim_exp, dim=-1, keepdim=False)
            fi_a = torch.sum(sim_exp, dim=-1, keepdim=False)
            zhi = mu_a/fi_a
            zhi = torch.where(zhi == 0, torch.tensor(1.0), zhi)
            mu_b = torch.sum(GT_sim*sim_exp, dim=1, keepdim=False)
            fi_b = torch.sum(sim_exp, dim=1, keepdim=False)
            zhi_2 = mu_b/fi_b
            zhi_2 = torch.where(zhi_2 == 0, torch.tensor(1.0), zhi_2)
            Align_Loss_1 = 0.001*(torch.sum(-torch.log(zhi + 1e-8))/sim_exp.shape[0] + torch.sum(-torch.log(zhi_2 + 1e-8))/sim_exp.shape[0])

            
        mask_sim = mask_from_lens(AV_attn.transpose(1,2), x_lengths, lip_lengths)
        alignment = maximum_path(AV_attn.transpose(1,2).contiguous(), mask_sim)

        pred_txt_un_phoneme = torch.matmul(alignment.transpose(1, 2), mu_x.transpose(1,2))

        x_fused = torch.cat([pred_txt_un_phoneme, output_text_lip], dim=-1)
        x_fused = self.layers_fused(x_fused)
        x_fused, _ = self.audio_visual_encoder(x_fused, lip_lengths)
        

        
        x_fused = torch.unsqueeze(x_fused.transpose(1,2), dim=1) 
        x_fused = F.leaky_relu(self.upsample_conv2d[0](x_fused), 0.4)  
        x_fused = F.leaky_relu(self.upsample_conv2d[1](x_fused), 0.4) 
        x_fused = torch.squeeze(x_fused, dim=1).transpose(1,2) 

        ctc_pred_mel = self.CTC_classifier_mel(x_fused)
        CTC_loss_MEL = 0.01*self.CTC_criterion(ctc_pred_mel.transpose(0, 1).log_softmax(2), x, y_lengths, x_lengths) / x.shape[0]

        
        mel_masks_formerDecoder = get_mask_from_lengths(y_lengths, y.shape[-1])
        pred_d_mu, _ = self.decoder_Condition(x_fused, mel_masks_formerDecoder, spks)
        
        pred_d_mu = self.mel_linear(pred_d_mu)
        postnet_output = self.postnet(pred_d_mu, spks) + pred_d_mu
        

  
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(torch.float32)
        
        postnet_output = postnet_output.transpose(1,2) * y_mask 

        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=postnet_output, spks=spks, cond=cond)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - postnet_output) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return prior_loss, diff_loss, CTC_loss_MEL, Align_Loss_1, energy_loss, pitch_loss_v





class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)




def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx



class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number




class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding="SAME"):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                                       if padding == "SAME"
                                       else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))

    def forward(self, xs, squeeze=False):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)
        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs.squeeze(-1) if squeeze else xs



class Predictor(PitchPredictor):
    pass


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, NameforStats):
        super(VarianceAdaptor, self).__init__()
        
        pitch_quantization = "linear" 
        self.NameforStats = NameforStats
        self.use_energy_embed = True # model_config["variance_embedding"]["use_energy_embed"]
        self.predictor_grad = 0.1 # model_config["variance_predictor"]["predictor_grad"]

        self.hidden_size = 256 # model_config["transformer"]["encoder_hidden"]
        self.filter_size = 256 # model_config["variance_predictor"]["filter_size"]
        self.predictor_layers = 2 # model_config["variance_predictor"]["predictor_layers"]
        self.dropout =  0.5 # model_config["variance_predictor"]["dropout"]
        self.ffn_padding = "SAME" # model_config["transformer"]["ffn_padding"]
        self.kernel = 5 # model_config["variance_predictor"]["predictor_kernel"]
        
        self.proj_1d = torch.nn.Conv1d(256, 256, 1)
        
        self.valence_attention = nn.MultiheadAttention(256, 1, dropout=0.2)

        if self.use_energy_embed:
            self.energy_feature_level = "frame_level"
            assert self.energy_feature_level in ["phoneme_level", "frame_level"]
            energy_quantization = "linear" 
            assert energy_quantization in ["linear", "log"]
            n_bins = 256 
            if self.NameforStats == 'Chem':
                with open(
                    os.path.join("configs/stats/stats_Chem.json")
                ) as f:
                    stats = json.load(f)
                    energy_min, energy_max = stats["energy"][:2] 
            if self.NameforStats == 'GRID':
                with open(
                    os.path.join("configs/stats/stats_GRID.json")
                ) as f:
                    stats = json.load(f)
                    energy_min, energy_max = stats["energy"][:2] 
            self.energy_predictor = Predictor(
                self.hidden_size,
                n_chans=self.filter_size,
                n_layers=self.predictor_layers,
                dropout_rate=self.dropout, odim=1,
                padding=self.ffn_padding, kernel_size=self.kernel)
            

            self.pitch_predictor = Predictor(
                self.hidden_size,
                n_chans=self.filter_size,
                n_layers=self.predictor_layers,
                dropout_rate=self.dropout, odim=1,
                padding=self.ffn_padding, kernel_size=self.kernel)
        
            
            if energy_quantization == "log":
                self.energy_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.energy_bins = nn.Parameter(
                    torch.linspace(energy_min, energy_max, n_bins - 1),
                    requires_grad=False,
                )
            self.energy_embedding = Embedding(n_bins, self.hidden_size, padding_idx=0)


        if self.NameforStats == 'Chem':
            with open(
                os.path.join("configs/stats/stats_Chem.json")
            ) as f:
                stats = json.load(f)
                pitch_min, pitch_max = stats["pitch"][:2]
        if self.NameforStats == 'GRID':
            with open(
                os.path.join("configs/stats/stats_GRID.json")
            ) as f:
                stats = json.load(f)
                pitch_min, pitch_max = stats["pitch"][:2] 

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
            
        self.pitch_embedding = nn.Embedding(
                    n_bins, 256
                )


    def get_energy_embedding(self, x, target, mask, control):
        x.detach() + self.predictor_grad * (x - x.detach())
        prediction = self.energy_predictor(x, squeeze=True)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    

    def get_pitch_embedding(self, x, target, mask, control):
        x.detach() + self.predictor_grad * (x - x.detach())
        prediction = self.pitch_predictor(x, squeeze=True)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding
    
    
    def forward(
        self,
        x,
        VAfeature,
        lip_mask,
        src_masks,
        energy_target=None,
        pitch_target = None,
        e_control=1.0,
    ):

        output_1 = x.transpose(1,2).clone()
        
        emotion_face = self.proj_1d(VAfeature.transpose(1,2))
        contextual, _ = self.valence_attention(query=output_1.permute(1,0,2),
                                                    key=emotion_face.permute(2,0,1), 
                                                    value=emotion_face.permute(2,0,1),
                                                    key_padding_mask=lip_mask)
        contextual = contextual.transpose(0,1)
        
        energy_prediction, energy_embedding = self.get_energy_embedding(
            contextual, energy_target, src_masks, e_control
        )
        
        output_1 += energy_embedding
        
        x = output_1.clone()
        
        output_2 = x.clone()
        
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, src_masks, e_control
        )
        output_2 += pitch_embedding
        
        x = output_2.clone()

        return (
            x,
            energy_prediction,
            pitch_prediction,
        )



class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



class LinearNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True, 
                 spectral_norm=False,
                 ):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)
        
        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out


class ConvNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True, 
                 spectral_norm=False,
                 ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)
        
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


class Conv1dGLU(nn.Module):
    '''
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2*out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn




class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len





class CTC_classifier_mel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, Dub):
        size = Dub.size()
        Dub = Dub.reshape(-1, size[2]).contiguous()
        Dub = self.classifier(Dub)
        return Dub.reshape(size[0], size[1], -1) 
        




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._VF as _VF
from torch.nn.modules.utils import _single, _pair, _triple


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, weight_init="default", bias_init="default"):
        super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        # Variational Noise
        self.noise = None
        self.vn_std = None

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, x):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise
            
        # Apply Weight
        x = F.linear(x, weight, self.bias)

        return x


class Conv1d(nn.Conv1d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1, 
        bias=True,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        padding="same", 
        channels_last=False,
        weight_init="default",
        bias_init="default",
        mask=None
    ):
        super(Conv1d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0 if isinstance(padding, str) else padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        if not isinstance(padding, str):
            padding = "valid"

        # Assert
        assert padding in ["valid", "same", "causal"]

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "same":

            self.pre_padding = nn.ConstantPad1d(
                padding=(
                    (self.kernel_size[0] - 1) // 2, # left
                    self.kernel_size[0] // 2 # right
                ), 
                value=0,
            )

        elif padding == "same-left": # Prioritize left context rather than right for even kernels, cause strided convolution with even kernel to be asymmetric!

            self.pre_padding = nn.ConstantPad1d(
                padding=(
                    self.kernel_size[0] // 2, 
                    (self.kernel_size[0] - 1) // 2
                ), 
                value=0
            )

        elif padding == "causal":

            self.pre_padding = nn.ConstantPad1d(
                padding=(
                    self.kernel_size[0] - 1, 
                    0
                ), 
                value=0
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        # Mask
        self.register_buffer("mask", mask)

    def forward(self, x):

        # Padding and Permute
        x = self.pre_padding(self.input_permute(x))

        # Mask Filter
        if self.mask != None:
            weight = self.weight * self.mask
        else:
            weight = self.weight

        # Apply Weight
        x = self._conv_forward(x, weight, self.bias)

        # Permute
        x = self.output_permute(x)

        return x

class Conv2d(nn.Conv2d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros',
        device=None, 
        dtype=None,

        padding="same", 
        channels_last=False,
        weight_init="default",
        bias_init="default",
        mask=None
    ):
        
        super(Conv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding if isinstance(padding, int) or isinstance(padding, tuple) else 0, 
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        if isinstance(padding, nn.Module):

            self.pre_padding = padding

        elif isinstance(padding, str):

            # Assert
            assert padding in ["valid", "same", "same-left"]

            # Padding
            if padding == "valid":

                self.pre_padding = nn.Identity()

            elif padding == "same":

                self.pre_padding = nn.ConstantPad2d(
                    padding=(
                        (self.kernel_size[1] - 1) // 2, # left
                        self.kernel_size[1] // 2, # right
                        
                        (self.kernel_size[0] - 1) // 2, # top
                        self.kernel_size[0] // 2 # bottom
                    ), 
                    value=0
                )

            elif padding == "same-left": # Prioritize left context rather than right for even kernels, cause strided convolution with even kernel to be asymmetric!

                self.pre_padding = nn.ConstantPad2d(
                    padding=(
                        self.kernel_size[1] // 2,
                        (self.kernel_size[1] - 1) // 2, 

                        self.kernel_size[0] // 2,
                        (self.kernel_size[0] - 1) // 2 
                    ), 
                    value=0
                )
        
        elif isinstance(padding, int) or isinstance(padding, tuple):
            
            self.pre_padding = nn.Identity()

        else:

            raise Exception("Unknown padding: ", padding, type(padding))

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        # Mask
        self.register_buffer("mask", mask)

    def forward(self, x):

        # Padding and Permute
        x = self.pre_padding(self.input_permute(x))

        # Mask Filter
        if self.mask != None:
            weight = self.weight * self.mask
        else:
            weight = self.weight

        # Apply Weight
        x = self._conv_forward(x, weight, self.bias)

        # Permute
        x = self.output_permute(x)

        return x

class Conv3d(nn.Conv3d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1, 
        bias=True,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        padding="same", 
        channels_last=False,
        weight_init="default",
        bias_init="default",
        mask=None
    ):

        super(Conv3d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding if isinstance(padding, int) or isinstance(padding, tuple) else 0,
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        if isinstance(padding, nn.Module):

            self.pre_padding = padding

        elif isinstance(padding, str):   

            # Assert
            assert padding in ["valid", "same", "same-left", "causal", "replicate"]

            # Padding
            if padding == "valid":

                self.pre_padding = nn.Identity()

            elif padding == "same":

                self.pre_padding = nn.ConstantPad3d(
                    padding=(
                        (self.kernel_size[2] - 1) // 2, # left
                        self.kernel_size[2] // 2, # right
                        
                        (self.kernel_size[1] - 1) // 2, # top
                        self.kernel_size[1] // 2, # bottom
                        
                        (self.kernel_size[0] - 1) // 2, # front
                        self.kernel_size[0] // 2 # back
                    ), 
                    value=0
                )

            elif padding == "replicate":

                self.pre_padding = nn.ReplicationPad3d(
                    padding=(
                        (self.kernel_size[2] - 1) // 2, # left
                        self.kernel_size[2] // 2, # right
                        
                        (self.kernel_size[1] - 1) // 2, # top
                        self.kernel_size[1] // 2, # bottom
                        
                        (self.kernel_size[0] - 1) // 2, # front
                        self.kernel_size[0] // 2 # back
                    )
                )

            elif padding == "same-left": # Prioritize left context rather than right for even kernels, cause strided convolution with even kernel to be asymmetric!

                self.pre_padding = nn.ConstantPad3d(
                    padding=(
                        self.kernel_size[2] // 2,
                        (self.kernel_size[2] - 1) // 2, 

                        self.kernel_size[1] // 2,
                        (self.kernel_size[1] - 1) // 2, 

                        self.kernel_size[0] // 2,
                        (self.kernel_size[0] - 1) // 2 
                    ), 
                    value=0
                )

            elif padding == "causal":

                self.pre_padding = nn.ConstantPad3d(
                    padding=(
                        (self.kernel_size[2] - 1) // 2,
                        self.kernel_size[2] // 2,
                        
                        (self.kernel_size[1] - 1) // 2,
                        self.kernel_size[1] // 2,
                        
                        self.kernel_size[0] - 1,
                        0
                    ), 
                    value=0
                )

        elif isinstance(padding, int) or isinstance(padding, tuple):
            
            self.pre_padding = nn.Identity()

        else:

            raise Exception("Unknown padding: ", padding, type(padding))

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        # Mask
        if mask == "A":

            stem_kernel_numel = torch.prod(torch.tensor(self.kernel_size))
            mask = torch.cat([
                torch.ones(torch.div(stem_kernel_numel, 2, rounding_mode="floor"), dtype=torch.float32), 
                torch.zeros(torch.div(stem_kernel_numel + 1, 2, rounding_mode="floor"), dtype=torch.float32)
            ], dim=0).reshape(self.kernel_size)

        elif mask == "B":
            
            stem_kernel_numel = torch.prod(torch.tensor(self.kernel_size))
            mask = torch.cat([
                torch.ones(torch.div(stem_kernel_numel + 1, 2, rounding_mode="floor"), dtype=torch.float32), 
                torch.zeros(torch.div(stem_kernel_numel, 2, rounding_mode="floor"), dtype=torch.float32)
            ], dim=0).reshape(self.kernel_size)

        self.register_buffer("mask", mask)

    def forward(self, x):

        # Padding and Permute
        x = self.pre_padding(self.input_permute(x))

        # Mask Filter
        if self.mask != None:
            weight = self.weight * self.mask
        else:
            weight = self.weight

        # Apply Weight
        x = self._conv_forward(x, weight, self.bias)

        # Permute
        x = self.output_permute(x)

        return x

class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0,
        output_padding=0, 
        groups=1, 
        bias=True, 
        dilation=1,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        weight_init="default",
        bias_init="default",
        channels_last=False
    ):

        super(ConvTranspose1d, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                output_padding=output_padding, 
                groups=groups, 
                bias=bias,
                dilation=dilation,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

    def forward(self, x):

        # Apply Weight
        x = self.output_permute(super(ConvTranspose1d, self).forward(self.input_permute(x)))

        return x

class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0,
        output_padding=0, 
        groups=1, 
        bias=True, 
        dilation=1,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        channels_last=False,
        weight_init="default",
        bias_init="default"
    ):

        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            output_padding=output_padding, 
            groups=groups, 
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

    def forward(self, x):

        # Apply Weight
        x = self.output_permute(super(ConvTranspose2d, self).forward(self.input_permute(x)))

        return x

class ConvTranspose3d(nn.ConvTranspose3d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0,
        output_padding=0, 
        groups=1, 
        bias=True, 
        dilation=1,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        channels_last=False,
        weight_init="default",
        bias_init="default",
    ):
        super(ConvTranspose3d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=tuple([k - 1 for k in _triple(kernel_size)]) if isinstance(padding, str) else padding,
            output_padding=output_padding, 
            groups=groups, 
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        if not isinstance(padding, str):
            padding = "valid"

        # Assert
        assert padding in ["valid", "replicate"]

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "replicate":

            self.pre_padding = nn.ReplicationPad3d(
                padding=(
                    (self.kernel_size[2] - 1) // 2, # left
                    (self.kernel_size[2] - 1) // 2, # right
                    
                    (self.kernel_size[1] - 1) // 2, # top
                    (self.kernel_size[1] - 1) // 2, # bottom
                     
                    (self.kernel_size[0] - 1) // 2, # front
                    (self.kernel_size[0] - 1) // 2 # back
                )
            )

    def forward(self, x):

        return self.output_permute(super(ConvTranspose3d, self).forward(self.input_permute(self.pre_padding(x))))

###############################################################################
# Pooling Layers
###############################################################################

class MaxPool1d(nn.MaxPool1d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False,

        padding="same",
        channels_last=False
    ):

        super(MaxPool1d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

        # Assert
        paddings = ["valid", "same", "causal"]
        assert padding in paddings, "possible paddings are " + ", ".join(paddings)

        # Padding
        if padding == "valid":
            self.pre_padding = nn.Identity()
        elif padding == "same":
            self.pre_padding = nn.ConstantPad1d(padding=(self.kernel_size[0] // 2, (self.kernel_size[0] - 1) // 2), value=0)
        elif padding == "causal":
            self.pre_padding = nn.ConstantPad1d(padding=(self.kernel_size[0] - 1, 0), value=0)


        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):

        # Padding
        x = self.pre_padding(self.input_permute(x))

        # Apply Weight
        x = self.output_permute(super(MaxPool1d, self).forward(x))

        return x

class MaxPool2d(nn.MaxPool2d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False,

        padding="same",
        channels_last=False
    ):

        super(MaxPool2d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

        # Assert
        paddings = ["valid", "same"]
        assert padding in paddings, "possible paddings are " + ", ".join(paddings)

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "same":

            self.pre_padding = nn.ConstantPad2d(
                padding=(
                    self.kernel_size[1] // 2,
                    (self.kernel_size[1] - 1) // 2, 
                    self.kernel_size[0] // 2,
                    (self.kernel_size[0] - 1) // 2 
                ), 
                value=0
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):

        # Padding
        x = self.pre_padding(self.input_permute(x))

        # Apply Weight
        x = self.output_permute(super(MaxPool2d, self).forward(x))

        return x

class MaxPool3d(nn.MaxPool3d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False,

        padding="same",
        channels_last=False
    ):

        super(MaxPool3d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

        # Assert
        paddings = ["valid", "same", "causal"]
        assert padding in paddings, "possible paddings are " + ", ".join(paddings)

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "same":

            self.pre_padding = nn.ConstantPad3d(
                padding=(
                    self.kernel_size[2] // 2,
                    (self.kernel_size[2] - 1) // 2, 
                    self.kernel_size[1] // 2,
                    (self.kernel_size[1] - 1) // 2, 
                    self.kernel_size[0] // 2,
                    (self.kernel_size[0] - 1) // 2 
                ), 
                value=0
            )

        elif padding == "causal":

            self.pre_padding = nn.ConstantPad3d(
                padding=(
                    self.kernel_size[2] // 2,
                    (self.kernel_size[2] - 1) // 2, 
                    self.kernel_size[1] // 2,
                    (self.kernel_size[1] - 1) // 2, 
                    self.kernel_size[0] - 1,
                    0
                ), 
                value=0
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):

        # Padding
        x = self.pre_padding(self.input_permute(x))

        # Apply Weight
        x = self.output_permute(super(MaxPool3d, self).forward(x))

        return x

class AvgPool1d(nn.AvgPool1d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True,
        
        channels_last=False
    ):

        super(AvgPool1d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, 
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(AvgPool1d, self).forward(self.input_permute(x)))

class AvgPool2d(nn.AvgPool2d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True,
        
        channels_last=False
    ):

        super(AvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, 
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(AvgPool2d, self).forward(self.input_permute(x)))

class AvgPool3d(nn.AvgPool3d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True,
        
        channels_last=False
    ):

        super(AvgPool3d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, 
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(AvgPool3d, self).forward(self.input_permute(x)))

class Upsample(nn.Upsample):

    def __init__(
        self,
        size=None, 
        scale_factor=None, 
        mode='nearest', 
        align_corners=None, 
        recompute_scale_factor=None,
        
        channels_last=False
    ):

        super(Upsample, self).__init__(
            size=size,
            scale_factor=scale_factor,
            mode=mode, 
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(to_last=False)
            self.output_permute = PermuteChannels(to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(Upsample, self).forward(self.input_permute(x)))

###############################################################################
# RNN Layers
###############################################################################

class LSTM(nn.LSTM):

    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional):
        super(LSTM, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=batch_first, 
            bidirectional=bidirectional)

        # Variational Noise
        self.noises = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noises = []
        for i in range(0, len(self._flat_weights), 4):
            self.noises.append(torch.normal(mean=0.0, std=1.0, size=self._flat_weights[i].size(), device=self._flat_weights[i].device, dtype=self._flat_weights[i].dtype))
            self.noises.append(torch.normal(mean=0.0, std=1.0, size=self._flat_weights[i+1].size(), device=self._flat_weights[i+1].device, dtype=self._flat_weights[i+1].dtype))

        # Broadcast Noise
        if distributed:
            for noise in self.noises:
                torch.distributed.broadcast(noise, 0)

    def forward(self, input, hx=None):  # noqa: F811

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        # Add Noise
        if self.noises is not None and self.training:
            weight = []
            for i in range(0, len(self.noises), 2):
                weight.append(self._flat_weights[2*i] + self.vn_std * self.noises[i])
                weight.append(self._flat_weights[2*i+1] + self.vn_std * self.noises[i+1])
                weight.append(self._flat_weights[2*i+2])
                weight.append(self._flat_weights[2*i+3])
        else:
            weight = self._flat_weights

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, weight, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, weight, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            output_packed = nn.utils.rnn.PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


class Embedding(nn.Embedding): 

    def __init__(self, num_embeddings, embedding_dim, padding_idx = None, weight_init="default"):
        super(Embedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise

        # Apply Weight
        return F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

###############################################################################
# Regularization Layers
###############################################################################

class Dropout(nn.Dropout):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__(p, inplace)

    def forward(self, x):

        if self.p > 0:
            return F.dropout(x, self.p, self.training, self.inplace)
        else: 
            return x

###############################################################################
# Tensor Manipulation Layers
###############################################################################

class PermuteChannels(nn.Module):

    """ Permute Channels

    Channels_last to channels_first / channels_first to channels_last
    
    """

    def __init__(self, to_last=True, num_dims=None, make_contiguous=False):
        super(PermuteChannels, self).__init__()

        # To last
        self.to_last = to_last

        # Set dims
        if num_dims != None:
            self.set_dims(num_dims)
        else:
            self.dims = None

        # Make Contiguous
        self.make_contiguous = make_contiguous

    def set_dims(self, num_dims):

        if self.to_last:
            self.dims = (0,) + tuple(range(2, num_dims + 2)) + (1,)
        else:
            self.dims = (0, num_dims + 1) + tuple(range(1, num_dims + 1))

    def forward(self, x):

        if self.dims == None:
            self.set_dims(num_dims=x.dim()-2)

        x = x.permute(self.dims)

        if self.make_contiguous:
            x = x.contiguous()

        return x

class Upsample3d(nn.Upsample):

    def __init__(self, scale_factor):

        # Assert
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor, scale_factor)
        else:
            assert isinstance(scale_factor, list) or isinstance(scale_factor, tuple)
            assert len(scale_factor) == 3

        # Init
        super(Upsample3d, self).__init__(scale_factor=scale_factor)

class Flatten(nn.Flatten):

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x):

        return super(Flatten, self).forward(x)

class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):

        return x.transpose(self.dim0, self.dim1)

class Permute(nn.Module):

    def __init__(self, dims, make_contiguous=False):
        super(Permute, self).__init__()
        self.dims = dims
        self.make_contiguous = make_contiguous

    def forward(self, x):
        x = x.permute(self.dims)
        if self.make_contiguous:
            x = x.contiguous()
        return x

class Reshape(nn.Module):

    def __init__(self, shape, include_batch=True):
        super(Reshape, self).__init__()
        self.shape = tuple(shape)
        self.include_batch = include_batch

    def forward(self, x):

        if self.include_batch:
            return x.reshape(self.shape)
        else:
            return x.reshape(x.size()[0:1] + self.shape)

class Unsqueeze(nn.Module):

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):

        return x.unsqueeze(dim=self.dim)

class GlobalAvgPool1d(nn.Module):

    def __init__(self, dim=1, keepdim=False):
        super(GlobalAvgPool1d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, mask=None):

        if mask != None:
            x = (x * mask).sum(dim=self.dim, keepdim=self.keepdim) / mask.count_nonzero(dim=self.dim)
        else:
            x = x.mean(dim=self.dim, keepdim=self.keepdim)

        return x

class GlobalAvgPool2d(nn.Module):

    def __init__(self, dim=(2, 3), keepdim=False):
        super(GlobalAvgPool2d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, mask=None):

        if mask != None:
            x = (x * mask).sum(dim=self.dim, keepdim=self.keepdim) / mask.count_nonzero(dim=self.dim)
        else:
            x = x.mean(dim=self.dim, keepdim=self.keepdim)

        return x

class GlobalMaxPool2d(nn.Module):

    def __init__(self, dim=(2, 3), keepdim=False):
        super(GlobalMaxPool2d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, output_dict=False):

        x = x.amax(dim=self.dim, keepdim=self.keepdim)

        return x

class GlobalAvgPool3d(nn.Module):

    def __init__(self, axis=(2, 3, 4), keepdim=False):
        super(GlobalAvgPool3d, self).__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, x, mask=None):

        return x.mean(axis=self.axis, keepdim=self.keepdim)

###############################################################################
# Layer Dictionary
###############################################################################

layer_dict = {
    "Linear": Linear,

    "Conv1d": Conv1d,
    "Conv2d": Conv2d,
    "Conv3d": Conv3d,

    "ConvTranspose1d": ConvTranspose1d,
    "ConvTranspose2d": ConvTranspose2d,
    "ConvTranspose3d": ConvTranspose3d,

    "MaxPool3d": MaxPool3d,

    "Dropout": Dropout,

    "Flatten": Flatten,
    "Transpose": Transpose,
    "Permute": Permute,
    "Reshape": Reshape,
    "Unsqueeze": Unsqueeze,
    "GlobalAvgPool1d": GlobalAvgPool1d,
    "GlobalAvgPool2d": GlobalAvgPool2d,
    "GlobalAvgPool3d": GlobalAvgPool3d,
    "GlobalMaxPool2d": GlobalMaxPool2d
}



import math

# PyTorch
import torch.nn.init as init

def normal(tensor, mean=0.0, std=1.0):
    return init.normal_(tensor, mean=mean, std=std)

# U(a, b)
def uniform(tensor, a=0.0, b=1.0):
    return init.uniform_(tensor, a=a, b=b)

# U(-b, b) where b = sqrt(1/fan_in)
def scaled_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, a=math.sqrt(5), mode=mode)

# N(0, std**2) where std = sqrt(1/fan_in)
def scaled_normal_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, nonlinearity="linear", mode=mode)

# U(-b, b) where b = sqrt(3/fan_in)
def lecun_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, nonlinearity="linear", mode=mode)

# N(0, std**2) where std = sqrt(1/fan_in)
def lecun_normal_(tensor, mode="fan_in"):
    return init.kaiming_normal_(tensor, nonlinearity="linear", mode=mode)

# U(-b, b) where b = sqrt(6/fan_in)
def he_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, mode=mode)

# N(0, std**2) where std = sqrt(2/fan_in)
def he_normal_(tensor, mode="fan_in"):
    return init.kaiming_normal_(tensor, mode=mode)

# U(-b, b) where b = sqrt(6/(fan_in + fan_out))
def xavier_uniform_(tensor):
    return init.xavier_uniform_(tensor)

# N(0, std**2) where std = sqrt(2/(fan_in + fan_out))
def xavier_normal_(tensor):
    return init.xavier_normal_(tensor)

# N(0.0, 0.02)
def normal_02_(tensor):
    return init.normal_(tensor, mean=0.0, std=0.02)

init_dict = {
    "uniform": init.uniform_,
    "normal": init.normal_,

    "ones": init.ones_,
    "zeros": init.zeros_,

    "scaled_uniform": scaled_uniform_,
    "scaled_normal": scaled_normal_,

    "lecun_uniform": lecun_uniform_,
    "lecun_normal": lecun_normal_,

    "he_uniform": he_uniform_,
    "he_normal": he_normal_,

    "xavier_uniform": xavier_uniform_,
    "xavier_normal": xavier_normal_,

    "normal_02": normal_02_
}



import torch
import torch.nn as nn


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class ReLU(nn.ReLU):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace=inplace)

    def forward(self, x):
        return super(ReLU, self).forward(x)

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()

class TanhGLU(nn.Module):
    
    def __init__(self, dim):
        super(TanhGLU, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in.tanh() * x_gate.sigmoid()

###############################################################################
# Activation Function Dictionary
###############################################################################

act_dict = {
    None: Identity,
    "Identity": Identity,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "Tanh": nn.Tanh,
    "ReLU": ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GLU": nn.GLU,
    "Swish": Swish,
    "GELU": nn.GELU
}



###############################################################################
# Networks
###############################################################################

class ResNet(nn.Module):

    """ ResNet (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

    Models: 224 x 224
    ResNet18: 11,689,512 Params
    ResNet34: 21,797,672 Params
    ResNet50: 25,557,032 Params
    ResNet101: 44,549,160 Params
    Resnet152: 60,192,808 Params

    Reference: "Deep Residual Learning for Image Recognition" by He et al.
    https://arxiv.org/abs/1512.03385

    """

    def __init__(self, dim_input=3, dim_output=1000, model="ResNet50", include_stem=True, include_head=True):
        super(ResNet, self).__init__()

        assert model in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

        if model == "ResNet18":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]
            bottleneck = False
        elif model == "ResNet34":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [3, 4, 6, 3]
            bottleneck = False
        elif model == "ResNet50":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 6, 3]
            bottleneck = True
        elif model == "ResNet101":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 23, 3]
            bottleneck = True
        elif model == "ResNet152":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 8, 36, 3]
            bottleneck = True

        self.stem = nn.Sequential(
            layers.Conv2d(in_channels=dim_input, out_channels=dim_stem, kernel_size=(7, 7), stride=(2, 2), weight_init="he_normal", bias=False),
            normalizations.BatchNorm2d(num_features=dim_stem),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ) if include_stem else nn.Identity()

        # Blocks
        self.blocks = nn.ModuleList()
        for stage_id in range(4):

            for block_id in range(num_blocks[stage_id]):

                # Projection Block
                if block_id == 0:
                    if stage_id == 0:
                        stride = (1, 1)
                        bottleneck_ratio = 1
                        in_features = dim_stem
                    else:
                        stride = (2, 2)
                        bottleneck_ratio = 2
                        in_features = dim_blocks[stage_id-1]
                # Default Block
                else:
                    stride = (1, 1)
                    in_features = dim_blocks[stage_id]
                    bottleneck_ratio = 4

                if bottleneck:
                    self.blocks.append(blocks.ResNetBottleneckBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        bottleneck_ratio=bottleneck_ratio,
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))
                else:
                    self.blocks.append(blocks.ResNetBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))

        # Head
        self.head = nn.Sequential(
            layers.GlobalAvgPool2d(),
            layers.Linear(in_features=dim_blocks[-1], out_features=dim_output, weight_init="he_normal", bias_init="zeros")
        ) if include_head else nn.Identity()

    def forward(self, x):

        # (B, Din, H, W) -> (B, D0, H//4, W//4)
        x = self.stem(x)

        # (B, D0, H//4, W//4) -> (B, D4, H//32, W//32)
        for block in self.blocks:
            x = block(x)

        # (B, D4, H//32, W//32) -> (B, Dout)
        x = self.head(x)

        return x

class Transformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention", "params":{"num_heads": 4, "weight_init": "normal_02", "bias_init": "zeros"}}, ff_ratio=4, emb_drop_rate=0.1, drop_rate=0.1, act_fun="GELU", pos_embedding=None, mask=None, inner_dropout=False, weight_init="normal_02", bias_init="zeros", post_norm=False):
        super(Transformer, self).__init__()

        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=emb_drop_rate)

        # Mask
        self.mask = mask

        # Transformer Blocks
        self.blocks = nn.ModuleList([blocks.TransformerBlock(
            dim_model=dim_model,
            ff_ratio=ff_ratio,
            att_params=att_params,
            drop_rate=drop_rate,
            inner_dropout=inner_dropout,
            act_fun=act_fun,
            weight_init=weight_init,
            bias_init=bias_init,
            post_norm=post_norm
        ) for block_id in range(num_blocks)])

        # LayerNorm
        self.layernorm = nn.LayerNorm(normalized_shape=dim_model) if not post_norm else nn.Identity()

    def forward(self, x, lengths=None):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)

        # Input Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # LayerNorm
        x = self.layernorm(x)

        return x

class ConformerInterCTC(nn.Module):

    def __init__(self, dim_model, num_blocks, interctc_blocks, vocab_size, loss_prefix="ctc", att_params={"class": "MultiHeadAttention", "num_heads": 4}, conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": 31}}, ff_ratio=4, drop_rate=0.1, pos_embedding=None, mask=None, conv_stride=1, batch_norm=True):
        super(ConformerInterCTC, self).__init__()

        # Inter CTC Params
        self.interctc_blocks = interctc_blocks
        self.loss_prefix = loss_prefix

        # Single Stage
        if isinstance(dim_model, int):
            dim_model = [dim_model]
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]
        
        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask

        # Conformer Stages
        i = 1
        self.conformer_blocks = nn.ModuleList()
        # self.interctc_modules = nn.ModuleList()
        for stage_id in range(len(num_blocks)):

            # Conformer Blocks
            for block_id in range(num_blocks[stage_id]):

                # Transposed Block
                transposed_block = "Transpose" in conv_params["class"]

                # Downsampling Block
                down_block = ((block_id == 0) and (stage_id > 0)) if transposed_block else ((block_id == num_blocks[stage_id] - 1) and (stage_id < len(num_blocks) - 1))

                # Block
                self.conformer_blocks.append(blocks.ConformerBlock(
                    dim_model=dim_model[stage_id - (1 if transposed_block and down_block else 0)],
                    dim_expand=dim_model[stage_id + (1 if not transposed_block and down_block else 0)],
                    ff_ratio=ff_ratio,
                    drop_rate=drop_rate,
                    att_params=att_params[stage_id - (1 if transposed_block and down_block else 0)] if isinstance(att_params, list) else att_params,
                    conv_stride=1 if not down_block else conv_stride[stage_id] if isinstance(conv_stride, list) else conv_stride,
                    conv_params=conv_params[stage_id] if isinstance(conv_params, list) else conv_params,
                    batch_norm=batch_norm
                ))

                i += 1

    def forward(self, x, lengths):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)
            
        # Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        interctc_outputs = {}
        j = 0
        for i, block in enumerate(self.conformer_blocks):

            x = block(x, mask=mask)

            # Strided Block
            if block.stride > 1:

                # Stride Mask (1 or B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if lengths is not None:
                    lengths = torch.div(lengths - 1, block.stride, rounding_mode='floor') + 1

        return x, lengths 

    
    