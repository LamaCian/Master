import torch
from transformers.models.seggpt.modeling_seggpt import patchify


class ViT(torch.nn.Module):
    ...

    def __init__(self, chw = (1,28,28), n_patches = 7):
        super(ViT, self).__init__()

        self.chw = chw
        self.n_patches = n_patches

        self.patch_dim = (chw[1]/n_patches, chw[2]/n_patches)
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])

        self.linear_lay = torch.nn.Linear(self.input_d, self.hidden_d)
        self.class_token = torch.nn.Parameter(torch.rand(self.input_id, self.hidden_d))
    def forward(self, images):
        patches = patchify(images, self.n_patches)
        tokens = self.linear_lay(patches)
        # Adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

    # We need to transform an img
    # Image - (N, C, H, W) -> (N, Patches, Patch Dim)]\
    # if we want to break 28x28 img into 7x7 patches
    # (N, P^2, HWC/P^2) -> (N, 7x7, 4x4) -> (N, 49, 16)_
    def patchify(self, images, n_patches):
        ...
        n,c, h, w = images.shape
        patches = torch.zeros(n, n_patches**2, h*w*c//n_patches**3)
        patch_dim = h // n_patches # 4

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_dim: (i + 1) * patch_dim, j * patch_dim: (j + 1) * patch_dim]
                    patches[idx, i * n_patches + j] = patch.flatten()

        return patches





