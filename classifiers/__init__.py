from classifiers.mlp import MLP
from classifiers.segnet import Segnet
from classifiers.small_convnet_mnist import SmallConvNetMNIST
from classifiers.small_convnet_svhn import SmallConvNetSVHN
from classifiers.vgg16 import VGG16

from classifiers.small_convnet_svhn_specnorm import SmallConvNetSVHN_SN

def get_base_classifier(classifier_name):
    """
        Return a new instance of model
    """
    
    if "resnet" in classifier_name:
        from classifiers import resnet_cifar, resnet_mnist
        model_factory = {
            "resnet50_cifar": resnet_cifar.ResNet50,
            "resnet18_cifar": resnet_cifar.ResNet18,
            "resnet18_mnist": resnet_mnist.ResNet18,

        }
        return model_factory[classifier_name]()

    else:
        # Available models
        model_factory = {
            "mlp": MLP,

            "small_convnet_mnist": SmallConvNetMNIST,

            "small_convnet_svhn": SmallConvNetSVHN,

            "vgg16": VGG16,

            "segnet": Segnet,
        }

        return model_factory[classifier_name](feature_dim=512)

