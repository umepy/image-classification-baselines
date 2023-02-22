import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def extract(pre_model: nn.Module, target, inputs):
    feature = None

    def forward_hook(module: nn.Module, inputs, outputs):
        global blocks
        blocks = outputs.detach()

    handle = target.register_forward_hook(forward_hook)

    # inference
    pre_model.eval()
    pre_model(inputs)

    handle.remove()
    return blocks


# TODO: need to define ViT class as model variable
def vit_attention_rollout(model: nn.Module, inputs: torch.Tensor):
    attention_weight = []

    for i in range(len(model.blocks)):
        target_module = model.blocks[i].attn
        features = extract(model, target_module, x)
        attention_weight.append([features.cpu().detach().numpy().copy()])
    attention_weight = np.squeeze(np.concatenate(attention_weight), axis=1)

    mean_head = np.mean(attention_weight, axis=1)
    mean_head = mean_head + np.eye(mean_head.shape[1])
    mean_head = mean_head / mean_head.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    v = mean_head[-1]
    for n in range(1, len(mean_head)):
        v = np.matmul(v, mean_head[-1 - n])

    mask = v[0, 1:].reshape(14, 14)
    attention_map = cv2.resize(mask / mask.max(), (ori_img.shape[2], ori_img.shape[3]))[..., np.newaxis]

    plt.imshow(attention_map)
    plt.savefig("./attention_rollout.png")


if __name__ == "__main__":
    model = Vit_model
