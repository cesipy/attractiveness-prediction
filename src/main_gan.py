import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random


from logger import Logger
import data_processor
from gan import BeautyGAN, GANDataset; import gan

logger = Logger()

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

    # data = data[:1000]
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
    gan: BeautyGAN = gan.load_pretrained_beauty_gan("checkpoints/models_only", device=device)

    for param_group in gan.opt_g.param_groups:
        param_group['lr'] = gan.LEARNING_RATE_G
    for param_group in gan.opt_d.param_groups:
        param_group['lr'] = gan.LEARNING_RATE_D


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