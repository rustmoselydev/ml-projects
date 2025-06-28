from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()  # Up to conv3_3
        for param in vgg_features.parameters():
            param.requires_grad = False

        self.vgg = vgg_features
        self.resize = resize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, input, target):
        # Both input and target assumed to be in [-1, 1], so denormalize first
        input = (input + 1) / 2
        target = (target + 1) / 2

        # If grayscale-ish, replicate channels to match VGG expectations
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Resize to 224x224 if needed
        if self.resize:
            input = nn.functional.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        input = self.normalize(input)
        target = self.normalize(target)

        input_features = self.vgg(input)
        target_features = self.vgg(target)

        return nn.functional.l1_loss(input_features, target_features)
