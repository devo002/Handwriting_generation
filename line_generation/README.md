# Line Generation

The purpose of this work is to generate handwriting in the style of any given writer and integrate the process into an interactive web application. Users can choose between two generation modes: the GAN-based generator or the pretrained Emuru model.

For the GAN generator, users can select a writer style from a dropdown menu, which automatically loads the corresponding writer's embedding to generate handwriting in that style. Alternatively, users can upload a reference image to mimic a specific writing style.
For the Emuru generator, the cached pretrained model is loaded and used to synthesize handwriting based on a given style.

The generated handwriting images can also be downloaded directly from the web interface.

![Architecture](coverimage.png)


## Reproduceability instructions


**Clone the repository and navigate to the directory**
```
git clone https://github.com/devo002/Handwriting_generation.git
cd line generation
```

**Create the conda environment from the requirements.txt and activate**
```
conda env create -n myenv python 3.10
conda activate myenv
pip install -r requirements.txt
```

## For training
In the configs directory are several jsons which have the parameters used for the paper. The "data_loader": "data_dir" needs set to the location of the dataset directory. You can also adjust the GPU options here.

First the handwriting recognition model and feature encoder networks need to be trained.
```
HWR: python train.py -c configs/cf_IAM_hwr_cnnOnly_batchnorm_aug.json
Encoder: python train.py -c configs/cf_IAM_auto_2tight_newCTC.json
Generator: python train.py -c configs/cf_IAMslant_noMask_charSpecSingleAppend_GANMedMT_autoAEMoPrcp2tightNewCTCUseGen_balB_hCF0.75_sMG.json
```

## Resume training from checkpoints
You can resume training from a previously saved checkpoint by
```
python train.py --resume path/to/checkpoint
```


## Generate styles
To generate the style embeddings 
```
python generate.py -c path/to/snapshot.pth -d output_directory -s style_pickle -g#[optional gpu flag]
```

## To download the Emuru model
```
from huggingface_hub import snapshot_download

# Download the Emuru model to the local cache (~/.cache/huggingface by default)
snapshot_download(repo_id="blowing-up-groundhogs/emuru")
```

## Setup app

Go to the app.py and change the directory of 

```
CKPT = /path/to/generatorssavedcheckpoint.pth
STYLE_PKL = /path/to/style_pickle
CHARSET_JSON = /path/to/charset.json
```

Then Run
```
streamlit run app.py
```

## Further Research

For additional details, you can refer to the following repositories below:

- [Handwriting Line Generation (herobd)](https://github.com/herobd/handwriting_line_generation/tree/master)
- [Emuru: Autoregressive Text-to-Image Generation (AIMAGELAB)](https://github.com/aimagelab/Emuru-autoregressive-text-img)





