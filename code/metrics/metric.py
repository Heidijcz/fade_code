import os
import pyiqa.utils.download_util as du
import pyiqa.archs.niqe_arch as na

LOCAL_NIQE_PATH = R"C:\Users\Administrator\.cache\torch\hub\checkpoints\niqe_modelparameters.mat"

def mock_load_file_from_url(url, *args, **kwargs):
    return LOCAL_NIQE_PATH
du.load_file_from_url = mock_load_file_from_url
na.load_file_from_url = mock_load_file_from_url

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import pyiqa

def psnr(img1, img2):
    metric = PeakSignalNoiseRatio(data_range=1.0).to(img1.device)
    return metric(img1, img2)

def ssim(img1, img2):
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
    return metric(img1, img2)

_niqe_instance = None

def niqe(img):
    global _niqe_instance
    if pyiqa is None:
        raise ImportError("NIQE metric requires 'pyiqa'. Please install it using: pip install pyiqa")

    if _niqe_instance is None:
        _niqe_instance = pyiqa.create_metric('niqe', device=img.device)

    if _niqe_instance.device != img.device:
        _niqe_instance = _niqe_instance.to(img.device)

    return _niqe_instance(img)