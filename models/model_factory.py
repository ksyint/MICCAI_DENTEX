"""
Model Factory for creating different architectures
Supports ResNet, EfficientNet, VGG with CBAM and Guided Attention
"""

from .resnet_cbam_guided import (
    resnet18_cbam_guided, resnet34_cbam_guided, resnet50_cbam_guided,
    resnet101_cbam_guided, resnet152_cbam_guided
)
from .efficientnet_cbam_guided import (
    efficientnet_b0_cbam_guided, efficientnet_b1_cbam_guided,
    efficientnet_b2_cbam_guided, efficientnet_b3_cbam_guided,
    efficientnet_b4_cbam_guided
)
from .vgg_cbam_guided import (
    vgg11_bn_cbam_guided, vgg13_bn_cbam_guided,
    vgg16_bn_cbam_guided, vgg19_bn_cbam_guided
)


MODEL_REGISTRY = {
    'resnet18': resnet18_cbam_guided,
    'resnet34': resnet34_cbam_guided,
    'resnet50': resnet50_cbam_guided,
    'resnet101': resnet101_cbam_guided,
    'resnet152': resnet152_cbam_guided,
    
    'efficientnet_b0': efficientnet_b0_cbam_guided,
    'efficientnet_b1': efficientnet_b1_cbam_guided,
    'efficientnet_b2': efficientnet_b2_cbam_guided,
    'efficientnet_b3': efficientnet_b3_cbam_guided,
    'efficientnet_b4': efficientnet_b4_cbam_guided,
    
    'vgg11_bn': vgg11_bn_cbam_guided,
    'vgg13_bn': vgg13_bn_cbam_guided,
    'vgg16_bn': vgg16_bn_cbam_guided,
    'vgg19_bn': vgg19_bn_cbam_guided,
}


def create_model(model_name, num_classes=1000, pretrained=False, 
                 use_cbam=True, guidance_weight=0.3, dropout=0.0, **kwargs):
    """
    Create a model with CBAM and Guided Attention
    
    Args:
        model_name: name of the model architecture
        num_classes: number of output classes
        pretrained: whether to load ImageNet pretrained weights
        use_cbam: whether to use CBAM modules
        guidance_weight: weight for bottom-region guidance (0-1)
        dropout: dropout rate
        **kwargs: additional arguments for specific models
    
    Returns:
        model: PyTorch model
    
    Example:
        >>> model = create_model('resnet50', num_classes=10, pretrained=True)
        >>> model = create_model('efficientnet_b0', num_classes=5, guidance_weight=0.4)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_fn = MODEL_REGISTRY[model_name]
    
    model_kwargs = {
        'num_classes': num_classes,
        'pretrained': pretrained,
        'use_cbam': use_cbam,
        'guidance_weight': guidance_weight,
    }
    
    if 'resnet' in model_name or 'vgg' in model_name:
        model_kwargs['dropout'] = dropout
    
    model_kwargs.update(kwargs)
    
    model = model_fn(**model_kwargs)
    
    return model


def get_available_models():
    """Return list of available model names"""
    return list(MODEL_REGISTRY.keys())


def print_model_info(model_name):
    """Print information about a specific model"""
    if model_name not in MODEL_REGISTRY:
        print(f"Model {model_name} not found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    if 'resnet' in model_name:
        print("Architecture: ResNet with CBAM and Guided Attention")
        print("Features:")
        print("  - Residual connections for better gradient flow")
        print("  - CBAM for channel and spatial attention")
        print("  - Guided spatial attention for bottom-region focus")
        
    elif 'efficientnet' in model_name:
        print("Architecture: EfficientNet with CBAM and Guided Attention")
        print("Features:")
        print("  - Mobile inverted bottleneck convolutions")
        print("  - Compound scaling (width, depth, resolution)")
        print("  - CBAM integrated into MBConv blocks")
        print("  - Guided spatial attention for bottom-region focus")
        
    elif 'vgg' in model_name:
        print("Architecture: VGG with CBAM and Guided Attention")
        print("Features:")
        print("  - Deep convolutional architecture")
        print("  - Batch normalization for stable training")
        print("  - CBAM after each conv block")
        print("  - Guided spatial attention for bottom-region focus")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    print("Available Models:")
    print("-" * 60)
    for i, name in enumerate(get_available_models(), 1):
        print(f"{i:2d}. {name}")
    
    print("\n" + "="*60)
    print("Example Model Creation:")
    print("="*60)
    
    model = create_model('resnet50', num_classes=10, pretrained=True, guidance_weight=0.3)
    print(f"✓ Created ResNet50 with 10 classes")
    
    model = create_model('efficientnet_b0', num_classes=5, use_cbam=True)
    print(f"✓ Created EfficientNet-B0 with 5 classes")
    
    model = create_model('vgg16_bn', num_classes=3, dropout=0.5)
    print(f"✓ Created VGG16-BN with 3 classes")
