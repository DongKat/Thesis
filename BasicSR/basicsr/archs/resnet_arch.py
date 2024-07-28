import cv2
import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import resnet as resnet

from basicsr.utils.registry import ARCH_REGISTRY

NAMES = {
    'resnet101': [
        'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'
    ]
}

@ARCH_REGISTRY.register()
class ResnetFeatureExtractor(nn.Module):
    def __init__(self,
                 layer_name_list,
                 resnet_type='resnet101',
                 resnet_weights=None,
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(ResnetFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[resnet_type]

        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if resnet_weights is not None:
            assert os.path.exists(resnet_weights), f'weights path {resnet_weights} does not exist.'
            resnet_model = getattr(resnet, resnet_type)(weights=None)
            resnet_model.fc = nn.Linear(2048, 5568)
            state_dict = torch.load(resnet_weights)
            resnet_model.load_state_dict(state_dict)
        else:
            # Load pretrained model from torchvision
            resnet_model = getattr(resnet, resnet_type)(pretrained=True)


        layers = list(resnet_model.children())[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, layers):
            modified_net[k] = v

        self.resnet_model = nn.Sequential(modified_net)


        # Freeze the model
        if not requires_grad:
            self.resnet_model.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.resnet_model.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.resnet_model._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output


# %%
# For debug
if __name__ == '__main__':
    opt = dict(
        layer_name_list=['layer4'],
        resnet_type='resnet101',
        resnet_weights=None,
        use_input_norm=True,
        range_norm=False,
        requires_grad=False,
        remove_pooling=False,
        pooling_stride=2
    )
    resnet_feature_extractor = ResnetFeatureExtractor(**opt)
    img = cv2.imread('baboon.png')
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
    x_features = resnet_feature_extractor(img_tensor)
    print(x_features.keys())

