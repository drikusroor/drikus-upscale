"""Stand-in for BiRefNet's build_backbone.py, reduced to the Swin backbones.

Upstream also wires up VGG/ResNet (torchvision), PVT and DINOv3, and loads
ImageNet weights for the backbone. None of that is needed to export: the
BiRefNet checkpoint already contains the trained backbone weights.
"""
from models.backbones.swin_v1 import swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l


def build_backbone(bb_name, pretrained=False, params_settings=''):
    return {'swin_v1_t': swin_v1_t, 'swin_v1_s': swin_v1_s,
            'swin_v1_b': swin_v1_b, 'swin_v1_l': swin_v1_l}[bb_name]()
