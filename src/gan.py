import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
from tqdm import tqdm

from config import *
import data_processor
from datasets import CustomDataset, GANDataset

LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 2e-5

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False 
torch.backends.cudnn.enabled = True

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
        
        # Generator architecture
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
    """Complete GAN system for beauty score interpolation"""
    
    def __init__(self, latent_dim=512, image_size=256, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        self.encoder = StyleGANEncoder(latent_dim, image_size).to(device)
        self.generator = StyleGANGenerator(latent_dim, image_size).to(device)
    
        self.discriminator = self._build_discriminator().to(device)
        

        self.opt_g = optim.Adam(list(self.encoder.parameters()) + list(self.generator.parameters()), 
                               lr=LEARNING_RATE_G, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(), 
                               lr=LEARNING_RATE_D, betas=(0.5, 0.999))

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        
        self.encoder = self.encoder.to(memory_format=torch.channels_last)
        self.generator = self.generator.to(memory_format=torch.channels_last)
        self.discriminator = self.discriminator.to(memory_format=torch.channels_last)
        
        # only works for python 3.n; n < 12 - had to downgrade
        self.encoder = torch.compile(self.encoder)
        self.generator = torch.compile(self.generator) 
        self.discriminator = torch.compile(self.discriminator)
        self.scaler = torch.cuda.amp.GradScaler()  
        
    def _build_discriminator(self):
        return nn.Sequential(
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
            nn.AdaptiveAvgPool2d(1),  # Force to 512 x 1 x 1
            nn.Flatten(),  # 512
            nn.Linear(512, 1),  # Output single value
        )
    
    def encode(self, images):
        return self.encoder(images)
    
    def decode(self, latents):
        return self.generator(latents)
    
    def train_step(self, real_images):
        
        real_images = real_images.to(memory_format=torch.channels_last)
        batch_size = real_images.size(0)
        with torch.cuda.amp.autocast():
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
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
            
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.opt_d)
        self.scaler.update()

        # autocast for mixed precision 
        with torch.cuda.amp.autocast():
            self.opt_g.zero_grad()
            fake_pred = self.discriminator(fake_images)
            g_adv_loss = self.adversarial_loss(fake_pred, real_labels)
            g_recon_loss = self.reconstruction_loss(fake_images, real_images)
            g_loss = g_adv_loss + 10.0 * g_recon_loss
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_recon_loss': g_recon_loss.item()
        }
    
    def train(self, dataloader, epochs=100):
        self.encoder.train()
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(epochs):
            epoch_losses = {'g_loss': 0, 'd_loss': 0, 'g_adv_loss': 0, 'g_recon_loss': 0}
            
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                images = images.to(self.device)
                
                losses = self.train_step(images)
                
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
            
            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader)
            
            print(f"Epoch {epoch+1}: G_loss: {epoch_losses['g_loss']:.4f}, "
                  f"D_loss: {epoch_losses['d_loss']:.4f}, "
                  f"Recon_loss: {epoch_losses['g_recon_loss']:.4f}")
            
            self.save_sample_images(dataloader, epoch+1)
    
    def interpolate_beauty_scores(self, face1, face2, score1, score2, num_interpolations=10):
        """
        Generate interpolated faces between two input faces with interpolated beauty scores
        
        Args:
            face1, face2: Input face tensors [C, H, W]
            score1, score2: Beauty scores for the faces
            num_interpolations: Number of interpolated samples to generate
            
        Returns:
            interpolated_faces: List of generated face tensors
            interpolated_scores: List of interpolated beauty scores
        """
        self.encoder.eval()
        self.generator.eval()
        
        with torch.no_grad():
            # make function more usable
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
                #interpolate in latent space, then decode
                interpolated_latent = alpha * latent1 + (1 - alpha) * latent2
                interpolated_face = self.decode(interpolated_latent)
                
                # do the same to the beatuy score
                interpolated_score = alpha * score1 + (1 - alpha) * score2
                
                interpolated_faces.append(interpolated_face.squeeze(0).cpu())
                interpolated_scores.append(interpolated_score)
        
        return interpolated_faces, interpolated_scores
    
    def save_sample_images(self, dataloader, epoch, save_dir="gan_samples"):
        """Save sample generated images after each epoch"""
        #genAI generated
        self.encoder.eval()
        self.generator.eval()
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            # Get a batch of real images
            real_batch = next(iter(dataloader))
            real_images = real_batch[0][:4].to(self.device)  # Take first 4 images
            
            # Encode and decode (reconstruction)
            latents = self.encode(real_images)
            reconstructed = self.decode(latents)
            
            # Create interpolation between first two images
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
        
        # Return to training mode
        self.encoder.train()
        self.generator.train()

class BeautyDatasetAugmenter:
    """Helper class to augment beauty dataset using GAN interpolation"""
    
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
            # Select face pairs based on strategy
            if score_pairs_strategy == 'extreme':
                # Pair high and low scoring faces
                idx1 = sorted_indices[np.random.randint(0, len(dataset)//4)]  # Low scoring
                idx2 = sorted_indices[np.random.randint(3*len(dataset)//4, len(dataset))]  # High scoring
            elif score_pairs_strategy == 'diverse':
                # Random pairs
                idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
            else:  # 'similar'
                # Pairs with similar scores
                base_idx = np.random.randint(len(dataset))
                window = len(dataset) // 10
                start = max(0, base_idx - window)
                end = min(len(dataset), base_idx + window)
                idx1 = sorted_indices[base_idx]
                idx2 = sorted_indices[np.random.randint(start, end)]
            
            face1, score1 = images[idx1], scores[idx1]
            face2, score2 = images[idx2], scores[idx2]
            
            # Generate interpolations
            interpolated_faces, interpolated_scores = self.gan.interpolate_beauty_scores(
                face1, face2, score1, score2, num_interpolations=5
            )
            
            # Add to augmented dataset (skip endpoints to avoid duplicates)
            for i in range(1, len(interpolated_faces)-1):
                if generated_count < num_to_generate:
                    augmented_images.append(interpolated_faces[i])
                    augmented_scores.append(interpolated_scores[i])
                    generated_count += 1
        
        return augmented_images, augmented_scores


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 256
    latent_dim = 768
    batch_size = 128
    epochs = 200
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    data_scut = data_processor.get_items_scut("res/data_scut", filter=DATASET_FILTER)
    avg_data = data_processor.get_averages(data=data_scut)
    
    data_me_train = data_processor.get_items_mebeauty("res/data_mebeauty/scores/train_cropped_scores.csv")
    # print(data_me_train[:5])
    data_me_test = data_processor.get_items_mebeauty("res/data_mebeauty/scores/test_cropped_scores.csv")
    data_thispersondoesnotexist = data_processor.get_items_thispersondoesnotexist("res/data_thispersondoesnotexist", fraction=0.4)
    data_celeba = data_processor.get_items_celeba(fraction=0.3)
    
    data = data_celeba + data_me_train + data_me_test + data_thispersondoesnotexist

    print(f"dataset size: {len(data)}")
    
    # thispersondoesnotexist_dir = "res/data_thispersondoesnotexist"
    # full_paths = []
    # for filename in os.listdir(thispersondoesnotexist_dir):
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         full_paths.append((os.path.join(thispersondoesnotexist_dir, filename), SCORE_PLACEHOLDER))
            
    # print("Total images found:", len(full_paths))
    # print(f"thispersondoesnotexist data: {full_paths[:5]}")
    
    # data = full_paths
    
    dataset = GANDataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)
    
    gan = BeautyGAN(latent_dim=latent_dim, image_size=image_size, device=device)
    

    print("Training GAN...")
    gan.train(dataloader, epochs=epochs)
    print("GAN training complete!")

    sample1 = dataset[0]
    sample2 = dataset[1]
    
    interpolated_faces, interpolated_scores = gan.interpolate_beauty_scores(
        sample1[0], sample2[0], sample1[1].item(), sample2[1].item(), num_interpolations=5
    )
    
    print(f"Generated {len(interpolated_faces)} interpolated faces")
    print(f"Score range: {min(interpolated_scores):.3f} - {max(interpolated_scores):.3f}")
    

if __name__ == "__main__":
    main()