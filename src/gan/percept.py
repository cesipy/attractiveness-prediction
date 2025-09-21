from torch import nn; import torch; from torch.nn import functional as F
import torchvision

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # simply compares loss of extracted features of pretrained model
        # really
        vgg = torchvision.models.vgg19(weights='IMAGENET1K_V1').features[:30]
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
