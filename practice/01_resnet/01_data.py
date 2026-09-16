import matplotlib.pyplot as plt

from data import PaperCIFAR10


dataset = PaperCIFAR10(train=True, augment=False)
image, label = dataset[0]
raw_image, _ = dataset.dataset[0]

print("dataset size:", len(dataset))
print("raw image size:", raw_image.size)
print("tensor shape:", image.shape)
print("tensor dtype:", image.dtype)
print("label:", label)
print("mean image shape:", dataset.mean_image.shape)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(raw_image)
axes[0].set_title("raw CIFAR-10")

restored = (image + dataset.mean_image).permute(1, 2, 0).clamp(0, 1)
axes[1].imshow(restored)
axes[1].set_title("tensor before mean subtraction")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
