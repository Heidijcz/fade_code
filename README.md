## Environment

Install dependencies:

```bash
conda create -n fade python=3.12

conda activate fade
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## Pretrained Weights

You can download the pretrained weights from:

[Google Drive](https://drive.google.com/file/d/1ONXqW54g_t7uuxogE5mmlVRbtfsPXPio/view?usp=drive_link)

After downloading, place the weight file in the `checkpoint/` directory, for example:

```text
fade_code/
|- checkpoint/
|  `- snow100k_best_model.pk
```

Then you can use the weight file in evaluation or inference:

- for evaluation, set `pre_trained_model` in [code/option/option_test.py](code/option/option_test.py) to the checkpoint path
- for demo / inference, pass the checkpoint path with `--weights`

## Dataset Layout

```text
datasets\Snow100K\
|- train/
|  |- synthetic/
|  |- mask/
|  |- gt/
|- test/
|   - Snow100K-L/
|     |- synthetic/
|     |- mask/
|     |- gt/
|   - Snow100K-M/
|     |- synthetic/
|     |- mask/
|     |- gt/
|   - Snow100K-S/
|     |- synthetic/
|     |- mask/
|     |- gt/
|- realistic/

datasets\CSD\
|- Train/
|  |- Snow/
|  |- Mask/
|  |- Gt/
|- Test/
|  |- Snow/
|  |- Mask/
|  |- Gt/

datasets\SRRS\
|- train/
|  |- Snow/
|  |- Mask/
|  |- Gt/
|- test/
|  |- Snow/
|  |- Mask/
|  |- Gt/
```

## Training

Run from the `code/` directory so relative output paths behave as expected:

```bash
cd code
python train.py
```

Training options are defined in [code/option/option_train.py](code/option/option_train.py).


## Evaluation

Run:

```bash
cd code
python test.py
```

Test options are in [code/option/option_test.py](code/option/option_test.py).

Make sure `pre_trained_model` in [code/option/option_test.py](code/option/option_test.py) points to your downloaded checkpoint file.

To save inferred images during evaluation, enable `--save_infer_results`:

```bash
cd code
python test.py --save_infer_results
```


## Demo / Inference

You can run inference on one image or an entire folder with [code/utils/demo.py](code/utils/demo.py):

```bash
python code/utils/demo.py --input image --output output --weights checkpoint/snow100k_best_model.pk
```

Useful arguments:

- `--input`: image file or folder
- `--output`: output file or folder
- `--weights`: checkpoint path
