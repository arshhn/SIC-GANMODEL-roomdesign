"""Simple Streamlit demo for the trained A-to-B room CycleGAN."""

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


IMAGE_SIZE = 256
WEIGHTS_PATH = Path(__file__).parent / "generator_A2B.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """This structure must match the generator used during Colab training."""

    def __init__(self, in_channels=3, out_channels=3, base=64, num_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base, 7),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        ]
        channels = base

        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        layers += [ResidualBlock(channels) for _ in range(num_blocks)]

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    channels, channels // 2, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_generator():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {WEIGHTS_PATH.name}. Upload it beside streamlit_app.py."
        )

    model = ResNetGenerator().to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


@torch.inference_mode()
def redesign(image, model):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    result = model(tensor)[0].cpu().clamp(-1, 1)
    result = (result + 1) / 2
    return transforms.ToPILImage()(result)


st.set_page_config(page_title="CycleGAN Room Styler", page_icon="🛋️", layout="centered")
st.title("🛋️ CycleGAN Room Styler")
st.write("Upload a plain room photo to create a designed-room version.")

uploaded_file = st.file_uploader(
    "Choose a room photo", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    if st.button("Redesign room", type="primary"):
        try:
            generator = load_generator()
            output_image = redesign(input_image, generator)
            left, right = st.columns(2)
            with left:
                st.image(input_image, caption="Original room", use_container_width=True)
            with right:
                st.image(output_image, caption="Redesigned room", use_container_width=True)
        except FileNotFoundError as error:
            st.error(str(error))
