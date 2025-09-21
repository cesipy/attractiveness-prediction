import torch;import torch.nn as nn;import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
from tqdm import tqdm

from datetime import datetime
import random

from config import *
import data_processor
from datasets import CustomDataset, GANDataset
from logger import Logger

from .encoder import Encoder
from .generator import Generator
from .percept import PerceptualLoss

logger = Logger()

LEARNING_RATE_G = 4e-5
LEARNING_RATE_D = 8e-5

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


class BeautyGAN:

    def __init__(
        self,
        latent_dim=1024,
        image_size=256,
        use_perceptual_loss=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):

        self.use_perceptual_loss = use_perceptual_loss
        self.device = device
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoder = Encoder(latent_dim, image_size).to(device)
        self.generator = Generator(latent_dim, image_size).to(device)
        self.discriminator = self._build_discriminator().to(device)

        self.encoder = torch.compile(self.encoder)
        self.generator = torch.compile(self.generator)
        self.discriminator = torch.compile(self.discriminator)

        self.opt_g = optim.Adam(list(self.encoder.parameters()) + list(self.generator.parameters()),
                               lr=LEARNING_RATE_G, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(),
                               lr=LEARNING_RATE_D, betas=(0.5, 0.999))

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        # if use_perceptual_loss:
        self.perceptual_loss = PerceptualLoss().to(device)

        self.encoder = self.encoder.to(memory_format=torch.channels_last)
        self.generator = self.generator.to(memory_format=torch.channels_last)
        self.discriminator = self.discriminator.to(memory_format=torch.channels_last)

        self.scaler = torch.cuda.amp.GradScaler()

    def _build_discriminator(self):
        from torch.nn.utils import spectral_norm

        return nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(512, 1)),
        )

    def r1_penalty(self, real_images):
        """R1 gradient penalty"""
        real_images.requires_grad_(True)
        real_pred = self.discriminator(real_images)
        grad_real = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_images, create_graph=True)[0]
        grad_penalty = (grad_real.reshape(grad_real.shape[0], -1).norm(dim=1) ** 2).mean()
        return grad_penalty



    def _get_smooth_labels(self, batch_size, real=True):
        if real:
            # Real labels: random between 0.8-1.0 (mostly confident, but not perfect)
            return torch.rand(batch_size, 1, device=self.device) * 0.2 + 0.8
        else:
            # Fake labels: random between 0.0-0.2 (mostly fake, but not absolutely certain)
            return torch.rand(batch_size, 1, device=self.device) * 0.2

    def train_step(self, real_images, batch_idx):
        real_images = real_images.to(memory_format=torch.channels_last)
        batch_size = real_images.size(0)


        with torch.cuda.amp.autocast():
            # smoothing labes to get more stable updates, suggested by genau
            real_labels = self._get_smooth_labels(batch_size, real=True)
            fake_labels = self._get_smooth_labels(batch_size, real=False)

            self.opt_d.zero_grad()

            # real images
            real_pred = self.discriminator(real_images)
            d_real_loss = self.adversarial_loss(real_pred, real_labels)

            # fake images
            latents, skip_connections = self.encoder(real_images)
            fake_images = self.generator(latents, skip_connections)
            fake_pred = self.discriminator(fake_images.detach())
            d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2

            if batch_idx % 32 == 0:
                gp = self.r1_penalty(real_images)
                d_loss += 5.0 * gp  # Reduced weight

        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.opt_d)
        self.scaler.update()

        # Train Generator
        for _ in range(2):  # Added this loop
            with torch.cuda.amp.autocast():
                self.opt_g.zero_grad()
                latents, skip_connections  = self.encoder(real_images)  # Re-encode for second iteration
                fake_images = self.generator(latents, skip_connections)
                fake_pred = self.discriminator(fake_images)
                g_adv_loss = self.adversarial_loss(fake_pred, real_labels)
                g_recon_loss = self.reconstruction_loss(fake_images, real_images)
                if self.use_perceptual_loss:
                    g_perceptual_loss = self.perceptual_loss(fake_images, real_images)
                    g_loss = 0.3 * g_adv_loss + 1.0 * g_recon_loss + 0.04 * g_perceptual_loss
                else:
                    g_loss = 0.3 * g_adv_loss + 1.0 * g_recon_loss

            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.opt_g)
            self.scaler.update()

        self.g_losses_buffer.append(g_loss.item())
        self.d_losses_buffer.append(d_loss.item())
        self.g_adv_losses_buffer.append(g_adv_loss.item())
        self.g_recon_losses_buffer.append(g_recon_loss.item())
        if self.use_perceptual_loss:
            self.perceptual_loss_buffer.append(g_perceptual_loss.item())


        if (batch_idx +1) % 10 == 0:
            avg_g_loss = sum(self.g_losses_buffer) / len(self.g_losses_buffer)
            avg_d_loss = sum(self.d_losses_buffer) / len(self.d_losses_buffer)
            avg_g_adv_loss = sum(self.g_adv_losses_buffer) / len(self.g_adv_losses_buffer)
            avg_g_recon_loss = sum(self.g_recon_losses_buffer) / len(self.g_recon_losses_buffer)
            if self.use_perceptual_loss:
                avg_g_perceptual_loss = sum(self.perceptual_loss_buffer) / len(self.perceptual_loss_buffer)


            self.g_losses_buffer = []
            self.d_losses_buffer = []
            self.g_adv_losses_buffer = []
            self.g_recon_losses_buffer = []
            if self.use_perceptual_loss:
                self.perceptual_loss_buffer = []

            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.g_adv_losses.append(avg_g_adv_loss)
            self.g_recon_losses.append(avg_g_recon_loss)
            if self.use_perceptual_loss:
                self.g_perceptual_losses.append(avg_g_perceptual_loss)


        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_recon_loss': g_recon_loss.item()
        } if not self.use_perceptual_loss else {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_recon_loss': g_recon_loss.item(),
            'g_perceptual_loss': g_perceptual_loss.item()
        }

    def train(self, dataloader, epochs=100,
            save_every=5, checkpoint_dir="checkpoints"
        ):
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()

        self.g_losses = []
        self.g_losses_buffer = []
        self.d_losses = []
        self.d_losses_buffer = []
        self.g_adv_losses = []
        self.g_adv_losses_buffer = []
        self.g_recon_losses = []
        self.g_recon_losses_buffer = []
        if self.use_perceptual_loss:
            self.g_perceptual_losses = []
            self.perceptual_loss_buffer = []


        for epoch in range(epochs):
            epoch_losses = {'g_loss': 0, 'd_loss': 0, 'g_adv_loss': 0, 'g_recon_loss': 0}
            if self.use_perceptual_loss:
                epoch_losses['g_perceptual_loss'] = 0

            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                images = images.to(self.device)

                losses = self.train_step(images, batch_idx)

                for key in epoch_losses:
                    epoch_losses[key] += losses[key]




            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader)

            if self.use_perceptual_loss:
                info_str = (f"Epoch {epoch+1}: G_loss: {epoch_losses['g_loss']:.4f}, "
                    f"D_loss: {epoch_losses['d_loss']:.4f}, "
                    f"Recon_loss: {epoch_losses['g_recon_loss']:.4f}, "
                    f"Perceptual_loss: {epoch_losses['g_perceptual_loss']:.4f}"
                )
            else:
                info_str = (
                    f"Epoch {epoch+1}: G_loss: {epoch_losses['g_loss']:.4f}, "
                    f"D_loss: {epoch_losses['d_loss']:.4f}, "
                    f"Recon_loss: {epoch_losses['g_recon_loss']:.4f}"
                )
            print(info_str)
            logger.info(info_str)

            if epoch % 3 == 0:
                self.save_sample_images(dataloader, epoch+1)

            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
                self.save_checkpoint(checkpoint_path, epoch=epoch+1, losses=epoch_losses)
                self.save_models_only(os.path.join(checkpoint_dir, "models_only"), epoch=epochs)

            # save visualization of loss
            plot_dir = "res/gan/plots"
            os.makedirs(plot_dir, exist_ok=True)
            self.plot_losses(save_dir=plot_dir)

        self.save_models_only(os.path.join(checkpoint_dir, "final_models"), epoch=epochs)

    def plot_losses(self, save_dir="checkpoints"):
        """Plot and save training losses"""
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))

        plt.plot(self.g_losses, label='Generator Loss', color='blue')
        plt.plot(self.d_losses, label='Discriminator Loss', color='red')
        plt.plot(self.g_recon_losses, label='Reconstruction Loss', color='green')
        if self.use_perceptual_loss and hasattr(self, 'g_perceptual_losses'):
            plt.plot(self.g_perceptual_losses, label='Perceptual Loss', color='orange')

        plt.xlabel('Batch (x10)')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(save_dir, 'training_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Loss plot saved to {save_dir}/training_losses.png")
        logger.info(f"Loss plot saved to {save_dir}/training_losses.png")


    def interpolate_beauty_scores(self, face1, face2, score1, score2, num_interpolations=10):
        self.encoder.eval()
        self.generator.eval()

        with torch.no_grad():
            if face1.dim() == 3:
                face1 = face1.unsqueeze(0)
            if face2.dim() == 3:
                face2 = face2.unsqueeze(0)

            face1 = face1.to(self.device)
            face2 = face2.to(self.device)

            latent1,skip_conns1 = self.encoder(face1)
            latent2,skip_conns2 = self.encoder(face2)

            interpolated_faces = []
            interpolated_scores = []

            alphas = np.linspace(0, 1, num_interpolations)

            for alpha in alphas:
                interpolated_latent = alpha * latent1 + (1 - alpha) * latent2
                interp_skips = [alpha * s1 + (1 - alpha) * s2 for s1, s2 in zip(skip_conns1, skip_conns2)]
                interpolated_face = self.generator(interpolated_latent, interp_skips)
                interpolated_score = alpha * score1 + (1 - alpha) * score2

                interpolated_faces.append(interpolated_face.squeeze(0).cpu())
                interpolated_scores.append(interpolated_score)

        return interpolated_faces, interpolated_scores

    def save_sample_images(self, dataloader, epoch, save_dir="gan_samples"):
        self.encoder.eval()
        self.generator.eval()

        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            real_batch = next(iter(dataloader))
            real_images = real_batch[0][:4].to(self.device)

            latents,skip_conns= self.encoder(real_images)
            reconstructed = self.generator(latents, skip_conns)

            latent1,skip_conns1 = self.encoder(real_images[0:1])
            latent2,skip_conns2 = self.encoder(real_images[1:2])

            interpolated_images = []
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            for alpha in alphas:
                interp_latent = alpha * latent1 + (1 - alpha) * latent2
                interp_skips = [alpha * s1 + (1 - alpha) * s2 for s1, s2 in zip(skip_conns1, skip_conns2)]
                interp_image = self.generator(interp_latent, interp_skips)
                interpolated_images.append(interp_image)

            # Convert from [-1, 1] to [0, 1] for saving
            def denormalize(tensor):
                return (tensor + 1) / 2

            # Save real images
            for i in range(4):
                real_img = denormalize(real_images[i])
                transforms.ToPILImage()(real_img.cpu()).save(
                    f"{save_dir}/epoch_{epoch:03d}_real_{i}.jpg"
                )

            # Save reconstructed images
            for i in range(4):
                recon_img = denormalize(reconstructed[i])
                transforms.ToPILImage()(recon_img.cpu()).save(
                    f"{save_dir}/epoch_{epoch:03d}_recon_{i}.jpg"
                )

            # Save interpolated images
            for i, interp_img in enumerate(interpolated_images):
                interp_img = denormalize(interp_img[0])
                transforms.ToPILImage()(interp_img.cpu()).save(
                    f"{save_dir}/epoch_{epoch:03d}_interp_{i}.jpg"
                )

            print(f"Saved sample images to {save_dir}/")

        self.encoder.train()
        self.generator.train()

    def save_checkpoint(self, checkpoint_path, epoch=None, losses=None, metadata=None):
            """Save complete model checkpoint"""
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_config': {
                    'latent_dim': self.latent_dim,
                    'image_size': self.image_size,
                },
                'encoder_state_dict': self.encoder.state_dict(),
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_g_state_dict': self.opt_g.state_dict(),
                'optimizer_d_state_dict': self.opt_d.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'losses': losses,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, load_optimizers=True):
        """Load complete model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)


        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if load_optimizers and 'optimizer_g_state_dict' in checkpoint:
            self.opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        losses = checkpoint.get('losses', {})
        metadata = checkpoint.get('metadata', {})

        print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
        return epoch, losses, metadata

    def save_models_only(self, save_dir, epoch=None):
        """Save only the trained models (for inference)"""
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.encoder.state_dict(), os.path.join(save_dir, 'encoder.pth'))
        torch.save(self.generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))


        config = {
            'latent_dim': self.latent_dim,
            'image_size': self.image_size,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(config, os.path.join(save_dir, 'config.pth'))

        print(f"Models saved to {save_dir}")
        logger.info(f"Models saved to {save_dir}")

    def load_models_only(self, save_dir):
        """Load only the trained models (for inference)"""

        config_path = os.path.join(save_dir, 'config.pth')
        if os.path.exists(config_path):
            config = torch.load(config_path, map_location=self.device)
            print(f"Loaded config from epoch {config.get('epoch', 'unknown')}")
            logger.info(f"Loaded config from epoch {config.get('epoch', 'unknown')}")

        self.encoder.load_state_dict(torch.load(os.path.join(save_dir, 'encoder.pth'), map_location=self.device))
        self.generator.load_state_dict(torch.load(os.path.join(save_dir, 'generator.pth'), map_location=self.device))
        self.discriminator.load_state_dict(torch.load(os.path.join(save_dir, 'discriminator.pth'), map_location=self.device))

        print(f"Models loaded from {save_dir}")
        logger.info(f"Models loaded from {save_dir}")



class BeautyDatasetAugmenter:
    def __init__(self, gan_model, beauty_model, device='cuda'):
        self.gan = gan_model
        self.beauty_model = beauty_model
        self.device = device

    def augment_dataset(self, dataset, target_size=None, score_pairs_strategy='extreme'):
        if target_size is None:
            target_size = len(dataset) * 2

        num_to_generate = target_size - len(dataset)

        augmented_images = []
        augmented_scores = []

        # Get images and scores
        images = [dataset[i][0] for i in range(len(dataset))]
        scores = [dataset[i][1] for i in range(len(dataset))]

        # Sort by beauty scores for strategic pairing
        sorted_indices = np.argsort(scores)

        generated_count = 0

        while generated_count < num_to_generate:
            if score_pairs_strategy == 'extreme':
                idx1 = sorted_indices[np.random.randint(0, len(dataset)//4)]
                idx2 = sorted_indices[np.random.randint(3*len(dataset)//4, len(dataset))]
            elif score_pairs_strategy == 'diverse':
                idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
            else:
                base_idx = np.random.randint(len(dataset))
                window = len(dataset) // 10
                start = max(0, base_idx - window)
                end = min(len(dataset), base_idx + window)
                idx1 = sorted_indices[base_idx]
                idx2 = sorted_indices[np.random.randint(start, end)]

            face1, score1 = images[idx1], scores[idx1]
            face2, score2 = images[idx2], scores[idx2]

            interpolated_faces, interpolated_scores = self.gan.interpolate_beauty_scores(
                face1, face2, score1, score2, num_interpolations=5
            )

            for i in range(1, len(interpolated_faces)-1):
                if generated_count < num_to_generate:
                    augmented_images.append(interpolated_faces[i])
                    augmented_scores.append(interpolated_scores[i])
                    generated_count += 1

        return augmented_images, augmented_scores


def load_pretrained_beauty_gan(model_dir, device='cuda'):
    """Load a pretrained BeautyGAN from saved models"""
    config_path = os.path.join(model_dir, 'config.pth')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = torch.load(config_path, map_location=device)

    # Create GAN with saved config
    gan = BeautyGAN(
        latent_dim=config['latent_dim'],
        image_size=config['image_size'],
        device=device
    )

    # Load the models
    gan.load_models_only(model_dir)

    return gan

def resume_training(checkpoint_path, dataloader, remaining_epochs, device='cuda'):
    """Resume training from a checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']

    # Create GAN
    gan = BeautyGAN(
        latent_dim=config['latent_dim'],
        image_size=config['image_size'],
        device=device
    )

    # Load checkpoint
    start_epoch, losses, metadata = gan.load_checkpoint(checkpoint_path)

    print(f"Resuming training from epoch {start_epoch}")
    print(f"Last losses: {losses}")

    for param_group in gan.opt_g.param_groups:
        param_group['lr'] = LEARNING_RATE_G
    for param_group in gan.opt_d.param_groups:
        param_group['lr'] = LEARNING_RATE_D

    print(f"Updated learning rates: G={LEARNING_RATE_G}, D={LEARNING_RATE_D}")
    logger.info(f"Updated learning rates: G={LEARNING_RATE_G}, D={LEARNING_RATE_D}")

    # gan.train(dataloader, epochs=remaining_epochs)

    return gan


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 256
    latent_dim = 1024
    batch_size = 80     # without perceptual loss
    # batch_size = 72
    epochs = 100

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.03, hue=0.05),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=2),
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # data_scut = data_processor.get_items_scut("res/data_scut", filter=DATASET_FILTER)
    # avg_data = data_processor.get_averages(data=data_scut)

    # data_me_train = data_processor.get_items_mebeauty("res/data_mebeauty/scores/train_cropped_scores.csv")
    # data_me_test = data_processor.get_items_mebeauty("res/data_mebeauty/scores/test_cropped_scores.csv")
    # data_thispersondoesnotexist = data_processor.get_items_thispersondoesnotexist("res/data_thispersondoesnotexist", fraction=0.9)
    # data_celeba = data_processor.get_items_celeba(fraction=1.)
    # data_celeba = data_celeba[:70000]


    # data = data_thispersondoesnotexist + data_me_train + data_me_test + data_celeba
    data = data_processor.get_items("res/gan/aligned_images")

    data = data[:1000]
    random.shuffle(data)
    print(f"dataset size: {len(data)}")
    logger.info(f"dataset size: {len(data)}")

    dataset = GANDataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True,
                            persistent_workers=False, prefetch_factor=4)

    gan = BeautyGAN(
        latent_dim=latent_dim,
        image_size=image_size,
        device=device,
        use_perceptual_loss=True
    )

    # print("Training GAN...")
    gan.train(dataloader, epochs=epochs)
    # print("GAN training complete!")

    # sample1 = dataset[0]
    # sample2 = dataset[1]

    # interpolated_faces, interpolated_scores = gan.interpolate_beauty_scores(
    #     sample1[0], sample2[0], sample1[1].item(), sample2[1].item(), num_interpolations=5
    # )

    # print(f"Generated {len(interpolated_faces)} interpolated faces")
    # print(f"Score range: {min(interpolated_scores):.3f} - {max(interpolated_scores):.3f}")


    # load the model
    gan: BeautyGAN = load_pretrained_beauty_gan("checkpoints/models_only", device=device)

    for param_group in gan.opt_g.param_groups:
        param_group['lr'] = LEARNING_RATE_G
    for param_group in gan.opt_d.param_groups:
        param_group['lr'] = LEARNING_RATE_D


    # gan = resume_training(
    #     checkpoint_path="checkpoints/checkpoint_epoch_065.pth",
    #     dataloader=dataloader,
    #     remaining_epochs=epochs,
    #     device=device
    # )

    # gan.use_perceptual_loss = False
    print("Pretrained GAN loaded successfully! now training on ")
    gan.train(dataloader, epochs=epochs, save_every=5, checkpoint_dir="checkpoints")

if __name__ == "__main__":
    main()