from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary
import argparse
import wandb
from torch.cuda.amp import GradScaler
import contextlib
from tqdm import tqdm
from asteroid.losses import pairwise_neg_sisdr
from asteroid.losses.pit_wrapper import PITLossWrapper

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs", type=int, default=20, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument(
    "--decay_epoch", type=int, default=50, help="epoch from which to start lr decay"
)
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument(
    "--cut_len",
    type=int,
    default=16000,
    help="cut length, default is 2 seconds in denoise " "and dereverberation",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="dir to VCTK-DEMAND dataset",
    help="dir of VCTK+DEMAND dataset",
)
parser.add_argument(
    "--save_model_dir", type=str, default="./saved_model", help="dir of saved model"
)
parser.add_argument(
    "--loss_weights",
    type=list,
    default=[0.0, 0.0, 0.0, 1.0],
    help="weights of RI components, magnitude, time loss, and Metric Disc",
)

parser.add_argument(
    "--overfit",
    action="store_true",
    help="overfit on single batch",
)

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def snr(est, target):
    return 20 * np.log10(
        np.linalg.norm(target) / (np.linalg.norm(target - est) + 1e-6) + 1e-6
    )


def si_sdr(est, target):
    alpha = (target * est).sum() / np.linalg.norm(target) ** 2
    return 20 * np.log10(
        np.linalg.norm(alpha * target) / (np.linalg.norm(alpha * target - est) + 1e-6)
        + 1e-6
    )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, train_ds, test_ds, overfit=False, use_mp=True):
        self.n_fft = 160
        self.hop = 100
        self.overfit = False
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.use_mp = use_mp and torch.cuda.is_available()
        self.scaler = GradScaler()
        self.model = TSCNet(num_channel=128, num_features=self.n_fft // 2 + 1).to(device)
        summary(
            self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)

        self.speaker_embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
                "auto_mix_prec": True,
            },
        )

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        """
        Move to utils
        """
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def forward_generator_step(self, s1, s2, mix):
        # Normalization

        mix_spec = torch.stft(
            mix,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
            return_complex=False,
        )
        s1_spec = torch.stft(
            s1,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
            return_complex=False,
        )
        s2_spec = torch.stft(
            s2,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
            return_complex=False,
        )

        mix_spec = power_compress(mix_spec).permute(0, 1, 3, 2)
        s1_spec = power_compress(s1_spec)
        s2_spec = power_compress(s2_spec)
        
        s1_real = s1_spec[:, 0, :, :].unsqueeze(1)
        s1_imag = s1_spec[:, 1, :, :].unsqueeze(1)

        s2_real = s2_spec[:, 0, :, :].unsqueeze(1)
        s2_imag = s2_spec[:, 1, :, :].unsqueeze(1)

        est_s1_real, est_s1_imag, est_s2_real, est_s2_imag,  = self.model(mix_spec, speaker_embed=None)
        est_s1_real, est_s1_imag = est_s1_real.permute(0, 1, 3, 2), est_s1_imag.permute(0, 1, 3, 2)
        est_s2_real, est_s2_imag = est_s2_real.permute(0, 1, 3, 2), est_s2_imag.permute(0, 1, 3, 2)

        est_s1_real = torch.sqrt(est_s1_real**2 + est_s1_imag**2)
        est_s2_real = torch.sqrt(est_s2_real**2 + est_s2_imag**2)

        s1_mag = torch.sqrt(s1_real**2 + s1_imag**2)
        s2_mag = torch.sqrt(s2_real**2 + s2_imag**2)

        est_spec_uncompress_s1 = power_uncompress(est_s1_real, est_s1_imag).squeeze(1)
        est_spec_uncompress_s2 = power_uncompress(est_s2_real, est_s2_imag).squeeze(1)

        est_audio_s1 = torch.istft(
            torch.view_as_complex(est_spec_uncompress_s1),
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
        )

        est_audio_s2 = torch.istft(
            torch.view_as_complex(est_spec_uncompress_s1),
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
        )

        return est_audio_s1, est_audio_s2, s1, s2

    def calculate_generator_loss(self, generator_outputs):

        s1_est, s2_est, s1, s2 = generator_outputs

        est_sources = torch.stack([s1_est, s2_est], dim=0).transpose(1,0)
        sources = torch.stack([s1, s2], dim=0).transpose(1,0)

        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        loss = loss_func(est_sources, sources)

        return loss

    def train_step(self, batch):
        # Trainer generator
        s1_ds, s2_ds, mix_ds, length = batch
        s1_ds, s2_ds, mix_ds, length = s1_ds.to(device), s2_ds.to(device), mix_ds.to(device), length.to(device)

        self.optimizer.zero_grad()

        mp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_mp
            else contextlib.suppress()
        )

        with mp_context:
            generator_outputs = self.forward_generator_step(
                s1_ds, s2_ds, mix_ds        
            )

            loss  = (
                self.calculate_generator_loss(generator_outputs)
            )

        if self.use_mp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return (
            loss.item(),
        )

    @torch.no_grad()
    def test_step(self, batch):
        s1_ds, s2_ds, mix_ds, length = batch
        s1_ds, s2_ds, mix_ds, length = s1_ds.to(device), s2_ds.to(device), mix_ds.to(device), length.to(device)
        
        mp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_mp
            else contextlib.suppress()
        )

        with mp_context:
            generator_outputs = self.forward_generator_step(
                s1_ds, s2_ds, mix_ds        
            )

            loss = self.calculate_generator_loss(generator_outputs)

        return loss.item()

    def test(self):
        self.model.eval()
        gen_loss_total = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(self.test_ds):
                step = idx + 1
                loss= self.test_step(batch)
                gen_loss_total += loss
            gen_loss_avg = gen_loss_total / max(step, 1)

        template = "GPU: {}, Validation loss: {}"
        wandb.log({"test/loss": gen_loss_avg})
        logging.info(template.format(0, gen_loss_avg))

        return gen_loss_avg

    def train(self):
        # scheduler_G = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=args.decay_epoch, gamma=0.5
        # )

        num_batches = len(self.train_ds)
        scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=args.epochs,
            steps_per_epoch=num_batches,
            pct_start=0.1,
            final_div_factor=100.0,
        )
        # scheduler_D = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        # )
        global_step = 0
        for epoch in range(args.epochs):
            self.model.train()
            for idx, batch in enumerate(self.train_ds):
                global_step += 1
                step = idx + 1
                loss = self.train_step(batch)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}"
                if (step % args.log_interval) == 0 or (
                    self.overfit and (epoch % args.log_interval == 0)
                ):
                    logging.info(template.format(0, epoch, step, loss))
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/lr": scheduler_G.get_last_lr()[0],
                            # "train/audio_source": wandb.Audio(
                            #     clean[0].numpy().T, sample_rate=8000
                            # ),
                            # "train/audio_est": wandb.Audio(
                            #     generated_audio[0].numpy().T, sample_rate=8000
                            # ),
                            # "train/audio_mix": wandb.Audio(
                            #     noisy[0].numpy().T, sample_rate=8000
                            # ),
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            # "train/loss_ri": loss_ri,
                            # "train/loss_mag": loss_mag,
                            # "train/time_loss": time_loss,
                            "model/grad_norm": self.get_grad_norm(self.model),
                        }
                    )
                scheduler_G.step()

            if not self.overfit:
                gen_loss = self.test()
                path = os.path.join(
                    args.save_model_dir,
                    "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
                )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            # if self.gpu_id == 0:
            if not self.overfit:
                torch.save(self.model.state_dict(), path)

            # scheduler_D.step()


def main(args):

    torch.backends.cudnn.benchmark = True

    wandb.login()

    run = wandb.init(
        project="mvoleynik_nemo",
        config={
            "init_lr": args.init_lr,
            "bath_size": args.batch_size,
            "loss_weights": args.loss_weights,
            "cut_len": args.cut_len,
            "decay_epoch": args.decay_epoch,
        },
    )
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, 4, args.cut_len, overfit=args.overfit
    )
    trainer = Trainer(train_ds, test_ds, overfit=args.overfit)
    trainer.train()


if __name__ == "__main__":
    main(args)
