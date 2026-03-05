import torch.nn as nn
import torchvision.models as tv_models


def get_model(model_name: str, num_classes: int):
    name = model_name.lower()

    if name == "alexnet":
        model = tv_models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        return model

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Supported in this artifact: alexnet."
    )