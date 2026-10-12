from .clip_models import CLIPModel, NoVIBCLIPModel
from .imagenet_models import ImagenetModel

VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',
    'CLIP:RN50',
    'CLIP:RN101',
    'CLIP:RN50x4',
    'CLIP:RN50x16',
    'CLIP:RN50x64',
    'CLIP:ViT-B/32',
    'CLIP:ViT-B/16',
    'CLIP:ViT-L/14',
    'CLIP:ViT-L/14@336px',
]


def get_model(
    name,
    classifier_head="linear",
    disable_vib=False,
    uncertainty_weighting=False,
    uncertainty_weight_scale=1.0,
    uncertainty_weight_mode="residual",
    vib_inference_mode="sample",
):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        if disable_vib:
            raise AssertionError("disable_vib is only supported for CLIP backbones in this project.")
        if uncertainty_weighting:
            raise AssertionError("uncertainty_weighting is only supported for CLIP-VIB models in this project.")
        return ImagenetModel(name[9:], classifier_head=classifier_head)
    if name.startswith("CLIP:"):
        if disable_vib:
            if uncertainty_weighting:
                raise AssertionError("uncertainty_weighting requires VIB and cannot be used with disable_vib.")
            return NoVIBCLIPModel(name[5:], classifier_head=classifier_head)
        return CLIPModel(
            name[5:],
            classifier_head=classifier_head,
            uncertainty_weighting=uncertainty_weighting,
            uncertainty_weight_scale=uncertainty_weight_scale,
            uncertainty_weight_mode=uncertainty_weight_mode,
            vib_inference_mode=vib_inference_mode,
        )
    raise AssertionError("Unsupported model name")
