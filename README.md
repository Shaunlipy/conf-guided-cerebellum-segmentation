# conf-guided-cerebellum-segmentation


## Data

[AB annotation](https://drive.google.com/file/d/1erZ_DVrG7JSRFbWvJrfoXnmN9DNvOVBz/view?usp=sharing)
files are in PNG format with integers from 0 to 5
- 0: tissue border
- 1: white matter
- 2: purkinje layer
- 3: granular layer
- 4: molecular layer
- 5: background

[AB annotation (RGB version for visualization)](https://drive.google.com/file/d/1G5L2QPWM9OtPt0EfNSnnv7YueRuvW4w-/view?usp=sharing)
- black: tissue border
- white: white matter
- red: purkinje layer
- green: granular layer
- blue: molecular layer
- grey: background

[AB input](https://drive.google.com/file/d/1-2wmDGPscTrnzg3Yj2lfwIXdAVWkV7B0/view?usp=sharing)
AB input patches

[BB input](https://drive.google.com/file/d/1-1gXOGcCS6uYgV9_O_hFPBj_5sqKMLWn/view?usp=sharing)
BB data input patches


## Training

To train the model.

<ol>
    <li> download and unzip all datasets (AB_input, AB_annotation, and BB_input) in to one root folder.
    <li> pre-train a segmentation model on the AB dataset.
    <li> generate pseudo labels for the BB.
    <li> jointly train AB and BB.
    <li> repeat 3-4 for iteratively self-training.
</ol>

Specifically:

for step 2, run:
```
python train.py --exp 0 --dataroot PathToYourDataroot
```

for step 3, run:
```angular2html
python inference.py --dataroot PathToYourDataroot --ckpt_dir output/0/model.pt
```

** replace 0/model.pth with corresponding training iteration (exp) when generating new pseudo labels after each iterative joint training **

for step 4, run:
```angular2html
python train.py --exp 1 --dataroot PathToYourDataroot
```

** replace --exp 1 with corresponding iteration **