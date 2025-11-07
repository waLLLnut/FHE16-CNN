import os, numpy as np, torch
from torchvision import datasets, transforms

save_dir = "mnist_batch"
os.makedirs(save_dir, exist_ok=True)

ds = datasets.MNIST("./data", train=False, download=True,
                    transform=transforms.ToTensor())

N = 1000
for i in range(N):
    img, label = ds[i]
    arr = (img.squeeze(0) * 255).round().to(torch.uint8).numpy()
    np.savetxt(os.path.join(save_dir, f"mnist_{i}.csv"), arr, fmt="%d", delimiter=",")
    with open(os.path.join(save_dir, f"mnist_{i}.label"), "w") as f:
        f.write(str(int(label)))
    if (i + 1) % 100 == 0:
        print(f"[Progress] Saved {i + 1}/{N}")

print(f"\n Saved {N} MNIST test samples to '{save_dir}/' directory.")

