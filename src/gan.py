import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
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

logger = Logger()

LEARNING_RATE_G = 4e-5  
LEARNING_RATE_D = 2.3e-5  

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False 
torch.backends.cudnn.enabled = True


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # simply compares loss of extracted features of pretrained model 
        # really 
        vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1').features[:25]
        self.vgg = vgg.eval()
        self.vgg = self.vgg.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = torch.compile(self.vgg)
        for p in self.vgg.parameters():
            p.requires_grad = False
    
    def forward(self, input, target):
        # VGG expects [0,1] normalized with ImageNet stats
        # images are [-1,1], so convert:
        input_norm = (input + 1) / 2  # [-1,1] -> [0,1]
        target_norm = (target + 1) / 2
        
        # image net norms
        # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input.device)
        
        input_norm = (input_norm - mean) / std
        target_norm = (target_norm - mean) / std
        
        input_features = self.vgg(input_norm)
        target_features = self.vgg(target_norm)
        return F.mse_loss(input_features, target_features)


class StyleGANEncoder(nn.Module):
    def __init__(self, latent_dim=512, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, 64, 4, 2, 1),  # 64 x 128 x 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1),  # 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1),  # 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 1, 0),  # 512 x 1 x 1
            nn.Flatten(),
            nn.Linear(512, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class StyleGANGenerator(nn.Module):
    """Simple generator to decode from latent space"""
    def __init__(self, latent_dim=512, image_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 512 x 16 x 16
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64 x 128 x 128
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 3 x 256 x 256
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.initial(z)
        x = x.view(-1, 512, 4, 4)
        return self.decoder(x)

class BeautyGAN:
    
    def __init__(
        self, 
        latent_dim=512, 
        image_size=256, 
        use_perceptual_loss=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        
        self.use_perceptual_loss = use_perceptual_loss
        self.device = device
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.encoder = StyleGANEncoder(latent_dim, image_size).to(device)
        self.generator = StyleGANGenerator(latent_dim, image_size).to(device)
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
    
    def encode(self, images):
        return self.encoder(images)
    
    def decode(self, latents):
        return self.generator(latents)
    
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
            # real_labels = torch.ones(batch_size, 1).to(self.device) 
            # fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # smoothing labes to get more stable updates.
            real_labels = self._get_smooth_labels(batch_size, real=True)
            fake_labels = self._get_smooth_labels(batch_size, real=False)
            
            self.opt_d.zero_grad()
            
            # Real images
            real_pred = self.discriminator(real_images)
            d_real_loss = self.adversarial_loss(real_pred, real_labels)
            
            # Fake images
            latents = self.encode(real_images)
            fake_images = self.decode(latents)
            fake_pred = self.discriminator(fake_images.detach())
            d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            # Add R1 penalty every 32 batches (less frequent)
            if batch_idx % 32 == 0:
                gp = self.r1_penalty(real_images)
                d_loss += 5.0 * gp  # Reduced weight
            
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.opt_d)
        self.scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            self.opt_g.zero_grad()
            fake_pred = self.discriminator(fake_images)
            g_adv_loss = self.adversarial_loss(fake_pred, real_labels)
            g_recon_loss = self.reconstruction_loss(fake_images, real_images)
            if self.use_perceptual_loss:

                g_perceptual_loss = self.perceptual_loss(fake_images, real_images)
            
                # reconstruction + small adversarial + conceptual
                g_loss = 0.1 * g_adv_loss + 1.0 * g_recon_loss + 0.04 * g_perceptual_loss    
            else: 
                g_loss = 0.1 * g_adv_loss + 1.0 * g_recon_loss
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()
        
        
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
                
        self.save_models_only(os.path.join(checkpoint_dir, "final_models"), epoch=epochs)

    
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
            
            latent1 = self.encode(face1)
            latent2 = self.encode(face2)
            
            interpolated_faces = []
            interpolated_scores = []
            
            alphas = np.linspace(0, 1, num_interpolations)
            
            for alpha in alphas:
                interpolated_latent = alpha * latent1 + (1 - alpha) * latent2
                interpolated_face = self.decode(interpolated_latent)
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
            
            latents = self.encode(real_images)
            reconstructed = self.decode(latents)
            
            latent1 = self.encode(real_images[0:1])
            latent2 = self.encode(real_images[1:2])
            
            interpolated_images = []
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            for alpha in alphas:
                interp_latent = alpha * latent1 + (1 - alpha) * latent2
                interp_image = self.decode(interp_latent)
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
        
        # Load model states
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states (optional)
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
        
        # Save individual models
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, 'encoder.pth'))
        torch.save(self.generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))
        
        # Save config
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
        # Load config
        config_path = os.path.join(save_dir, 'config.pth')
        if os.path.exists(config_path):
            config = torch.load(config_path, map_location=self.device)
            print(f"Loaded config from epoch {config.get('epoch', 'unknown')}")
            logger.info(f"Loaded config from epoch {config.get('epoch', 'unknown')}")
        
        # Load models
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
    
    gan.train(dataloader, epochs=remaining_epochs)
    
    return gan


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 256
    latent_dim = 512
    batch_size = 64
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
    
    data_scut = data_processor.get_items_scut("res/data_scut", filter=DATASET_FILTER)
    avg_data = data_processor.get_averages(data=data_scut)
    
    data_me_train = data_processor.get_items_mebeauty("res/data_mebeauty/scores/train_cropped_scores.csv")
    data_me_test = data_processor.get_items_mebeauty("res/data_mebeauty/scores/test_cropped_scores.csv")
    data_thispersondoesnotexist = data_processor.get_items_thispersondoesnotexist("res/data_thispersondoesnotexist", fraction=0.4)
    data_celeba = data_processor.get_items_celeba(fraction=1.)
    data_celeba = data_celeba[:10000]
    
    # Start with smaller dataset for testing
    #data = data_celeba + data_thispersondoesnotexist #data_me_train + data_me_test
    # data = data[4000]
    data = data_thispersondoesnotexist + data_me_train + data_me_test + data_celeba
    print(f"dataset size: {len(data)}")
    logger.info(f"dataset size: {len(data)}")
    
    dataset = GANDataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)
    
    gan = BeautyGAN(latent_dim=latent_dim, image_size=image_size, device=device)
    

    # print("Training GAN...")
    # gan.train(dataloader, epochs=epochs)
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
    
    gan.use_perceptual_loss = True
    print("Pretrained GAN loaded successfully! now training on ")
    gan.train(dataloader, epochs=epochs, save_every=5, checkpoint_dir="checkpoints")
    
if __name__ == "__main__":
    main()