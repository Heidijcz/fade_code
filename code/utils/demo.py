import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.backbone_train import fade

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def pad_img(x, patch_size):
    _, _, height, width = x.size()
    mod_pad_h = (patch_size - height % patch_size) % patch_size
    mod_pad_w = (patch_size - width % patch_size) % patch_size
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")


def parse_args():
    parser = argparse.ArgumentParser(description="Run fade desnowing on one image or a folder.")
    parser.add_argument(
        "--input",
        type=str,
        default=r"E:\Snowy2clear\fade_code\image",
        help="path of the snowy input image or input image folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"E:\Snowy2clear\fade_code\output",
        help="output image path for single-image mode, or output folder for folder mode",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=r"E:\Snowy2clear\fade_code\checkpoint\snow100k_best_model.pk",
        help="path of the trained fade weights",
    )
    parser.add_argument("--channel", type=int, default=40, help="fade base channel number")
    parser.add_argument("--pad-size", type=int, default=16, help="pad image to a multiple of this size")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively process images in subfolders when input is a folder",
    )
    return parser.parse_args()


def resolve_output_path(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / input_path.name
    if str(output_path).endswith((os.sep, "/")):
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def list_images(input_dir, recursive=False):
    pattern = "**/*" if recursive else "*"
    image_paths = [
        path
        for path in Path(input_dir).glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def load_checkpoint(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict):
        checkpoint = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in checkpoint.items()
        }
    model.load_state_dict(checkpoint)


@torch.no_grad()
def desnow_image(model, image_path, device, pad_size):
    image = Image.open(image_path).convert("RGB")
    tensor = ToTensor()(image).unsqueeze(0).to(device)
    height, width = tensor.shape[2:]

    if pad_size > 1:
        tensor = pad_img(tensor, pad_size)

    output, _ = model(tensor)
    output = output.clamp(0, 1)
    output = output[:, :, :height, :width]
    return ToPILImage()(output.squeeze(0).cpu())


def desnow_file(model, image_path, output_path, device, pad_size):
    result = desnow_image(model, image_path, device, pad_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def desnow_folder(model, input_dir, output_dir, device, pad_size, recursive=False):
    image_paths = list_images(input_dir, recursive=recursive)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Desnowing", unit="image"):
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        desnow_file(model, image_path, output_path, device, pad_size)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by this fade implementation because some modules create CUDA tensors directly."
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    weight_path = Path(args.weights)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model = fade(channel=args.channel).to(device).eval()
    load_checkpoint(model, weight_path, device)

    if input_path.is_dir():
        desnow_folder(model, input_path, output_path, device, args.pad_size, recursive=args.recursive)
        print(f"Saved desnowed images to: {output_path}")
    elif input_path.is_file():
        output_path = resolve_output_path(input_path, output_path)
        desnow_file(model, input_path, output_path, device, args.pad_size)
        print(f"Saved desnowed image to: {output_path}")
    else:
        raise FileNotFoundError(f"Input path is not a file or folder: {input_path}")


if __name__ == "__main__":
    main()
