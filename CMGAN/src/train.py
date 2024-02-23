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

import numpy as np

from torchmetrics import ScaleInvariantSignalNoiseRatio

from speechbrain.pretrained import EncoderClassifier


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
    default=[0.1, 0.2, 0.2, 0.5],
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
        self.overfit = overfit
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.use_mp = use_mp and torch.cuda.is_available()
        self.scaler = GradScaler()
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(device)
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
        self.speaker_embedder = self.speaker_embedder.to(device)

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

    def forward_generator_step(self, clean, noisy, reference):
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        se = self.speaker_embedder.encode_batch(reference.to(device))

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
            return_complex=False,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
            return_complex=False,
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec, speaker_embed=se)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            torch.view_as_complex(est_spec_uncompress),
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(device),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def calculate_generator_loss(self, generator_outputs):
        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = generator_outputs["est_audio"]
        clean_audio_list = generator_outputs["clean"][:, :length]
        gen_loss_GAN = -discriminator.sdr_loss(est_audio_list, clean_audio_list)

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            args.loss_weights[0] * loss_ri
            + args.loss_weights[1] * loss_mag
            + args.loss_weights[2] * time_loss
            + args.loss_weights[3] * gen_loss_GAN
        )

        return (
            loss,
            loss_ri.detach().cpu(),
            loss_mag.detach().cpu(),
            time_loss.detach().cpu(),
            gen_loss_GAN.detach().cpu(),
        )

    def train_step(self, batch):
        # Trainer generator
        clean = batch[0].to(device)
        noisy = batch[1].to(device)
        reference = batch[-1].to(device)

        self.optimizer.zero_grad()

        mp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_mp
            else contextlib.suppress()
        )

        with mp_context:
            generator_outputs = self.forward_generator_step(
                clean,
                noisy,
                reference,
            )
            generator_outputs["clean"] = clean

            sdr = si_sdr(
                generator_outputs["est_audio"].cpu().detach(),
                generator_outputs["clean"].cpu().detach(),
            )

            loss, loss_ri, loss_mag, time_loss, sdr_loss = (
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
            sdr,
            generator_outputs["est_audio"].cpu().detach(),
            clean.cpu(),
            noisy.cpu(),
            loss_ri,
            loss_mag,
            time_loss,
            sdr_loss,
        )

    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].to(device)
        noisy = batch[1].to(device)
        reference = batch[-1].to(device)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
            reference,
        )
        generator_outputs["clean"] = clean

        loss, _, _, _, sdr = self.calculate_generator_loss(generator_outputs)

        return loss.item(), sdr.cpu().item()

    def test(self):
        self.model.eval()
        gen_loss_total = 0.0
        sdr_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, sdr = self.test_step(batch)
            gen_loss_total += loss
            sdr_total += sdr
        gen_loss_avg = gen_loss_total / step
        sdr_total_avg = sdr_total / step

        template = "GPU: {}, Validation loss: {}, SDR: {}"
        wandb.log({"test/loss": gen_loss_avg, "test/snr": sdr_total_avg})
        logging.info(template.format(0, gen_loss_avg, sdr_total_avg))

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
            pct_start=10,
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
                (
                    loss,
                    snr,
                    generated_audio,
                    clean,
                    noisy,
                    loss_ri,
                    loss_mag,
                    time_loss,
                    sdr_loss,
                ) = self.train_step(batch)
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, snr: {}"
                if (step % args.log_interval) == 0 or (
                    self.overfit and (epoch % args.log_interval == 0)
                ):
                    logging.info(template.format(0, epoch, step, loss, snr))
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/snr": snr,
                            "train/lr": scheduler_G.get_last_lr()[0],
                            "train/audio_source": wandb.Audio(
                                clean[0].numpy().T, sample_rate=8000
                            ),
                            "train/audio_est": wandb.Audio(
                                generated_audio[0].numpy().T, sample_rate=8000
                            ),
                            "train/audio_mix": wandb.Audio(
                                noisy[0].numpy().T, sample_rate=8000
                            ),
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/loss_ri": loss_ri,
                            "train/loss_mag": loss_mag,
                            "train/time_loss": time_loss,
                            "model/grad_norm": self.get_grad_norm(self.model),
                        }
                    )

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
            scheduler_G.step()
            # scheduler_D.step()


def main(args):

    torch.backends.cudnn.benchmark = True

    wandb.login()

    run = wandb.init(
        project="cmgan-ss",
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
