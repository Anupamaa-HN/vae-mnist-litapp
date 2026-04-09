import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

st.set_page_config(page_title="VAE MNIST Explorer", page_icon="🧠", layout="wide")

# ── VAE Model ────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, 784),        nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = VAE(latent_dim=2)
    model.load_state_dict(torch.load('vae_model.pth', map_location='cpu'))
    model.eval()
    return model


# ── Load MNIST data (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=512, shuffle=False)


model = load_model()
loader = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 VAE MNIST Explorer")
st.sidebar.caption("Lab 1.2 — Ada Lovelace Software Pvt. Ltd")
page = st.sidebar.radio("Choose a view", [
    "📷 Reconstructions",
    "🗺️ Latent Space",
    "✨ Generate Digits",
    "🔀 Latent Interpolation"
])

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Reconstructions
# ════════════════════════════════════════════════════════════════════════════════
if page == "📷 Reconstructions":
    st.title("Original vs Reconstructed Images")
    st.caption("The VAE compresses each image into just 2 numbers, then rebuilds it.")

    n = st.slider("Number of images to show", 4, 16, 8)

    images, labels = next(iter(loader))
    images = images[:n]

    with torch.no_grad():
        recon, _, _ = model(images)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    for i in range(n):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(str(labels[i].item()), fontsize=9)
        axes[1, i].imshow(recon[i].view(28, 28).detach(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    plt.suptitle("Original vs Reconstructed", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.info("**Row 1** — Original MNIST images")
    col2.success("**Row 2** — VAE reconstructions (compressed to 2D and rebuilt)")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Latent Space
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Latent Space":
    st.title("2D Latent Space Visualisation")
    st.caption("Each dot is one image mapped to its 2D latent coordinate. Colour = digit class.")

    n_batches = st.slider("Batches to visualise (more = denser plot)", 1, 10, 5)

    z_all, labels_all = [], []
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            if i >= n_batches:
                break
            mu, _ = model.encode(data.view(-1, 784))
            z_all.append(mu.numpy())
            labels_all.append(labels.numpy())

    z = np.concatenate(z_all)
    lbls = np.concatenate(labels_all)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(z[:, 0], z[:, 1], c=lbls, cmap='tab10', alpha=0.5, s=6)
    plt.colorbar(sc, ax=ax, label='Digit class (0–9)')
    ax.set_xlabel("z₁ (latent dimension 1)")
    ax.set_ylabel("z₂ (latent dimension 2)")
    ax.set_title("VAE Latent Space — 2D Visualisation")
    plt.tight_layout()
    st.pyplot(fig)

    st.info("Each digit class naturally clusters together — the VAE learned structure without any labels during training!")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Generate Digits
# ════════════════════════════════════════════════════════════════════════════════
elif page == "✨ Generate Digits":
    st.title("Generate Brand New Digits")
    st.caption("Sample random points from the latent space → decode into new images.")

    col1, col2 = st.columns([1, 2])
    n_gen = col1.slider("How many to generate", 4, 32, 16)

    if st.button("✨ Generate!"):
        with torch.no_grad():
            z = torch.randn(n_gen, 2)
            imgs = model.decode(z)

        cols = 8
        rows = (n_gen + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
        for i, ax in enumerate(np.array(axes).flatten()):
            if i < n_gen:
                ax.imshow(imgs[i].view(28, 28).detach(), cmap='gray')
            ax.axis('off')
        plt.suptitle("Newly Generated Digits from Random Latent Vectors", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

        st.success("These digits have never existed before — generated purely from random noise!")
    else:
        st.info("Press Generate to create new digits from random latent vectors.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Latent Interpolation
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔀 Latent Interpolation":
    st.title("Digit Morphing — Latent Space Interpolation")
    st.caption("Smoothly morph between two random latent points. Watch one digit transform into another.")

    if "z1" not in st.session_state or st.button("🎲 New random pair"):
        st.session_state.z1 = torch.randn(1, 2)
        st.session_state.z2 = torch.randn(1, 2)

    steps = st.slider("Interpolation steps", 5, 20, 10)
    alphas = np.linspace(0, 1, steps)

    z1, z2 = st.session_state.z1, st.session_state.z2
    z_interp = torch.stack([(1 - a) * z1 + a * z2 for a in alphas]).squeeze(1)

    with torch.no_grad():
        imgs = model.decode(z_interp)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2.2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].view(28, 28).detach(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Start", fontsize=8)
        elif i == steps - 1:
            ax.set_title("End", fontsize=8)
    plt.suptitle("Smooth morphing through latent space →", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    st.info("This works because the VAE latent space is smooth and continuous — every point between two digits decodes into a valid image!")

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Built with PyTorch + Streamlit\nLab 1.2 | Generative AI Course")
