#  MIMIC-CXR Medical Entity Extraction Pipeline

## Basic Environment Setup (One time activity)
1. Clone the DYGIE++ repository from: https://github.com/dwadden/dygiepp. This repositiory is managed by Wadden et al., authors of the paper Entity, Relation, and Event Extraction with Contextualized Span Representations (https://www.aclweb.org/anthology/D19-1585.pdf).
```
git clone https://github.com/dwadden/dygiepp.git
```
2. Navigate to the root of repo in your system and use the following commands to setup the conda environment:

```
conda create --name dygiepp python=3.7
pip install -r requirements.txt
conda develop .   # Adds DyGIE to your PYTHONPATH
```

## Activate the conda environment:

```
conda activate dygiepp
```



## Run the inference.py file using the command:
```
python3 inference.py --model_path ./model.tar.gz \
 --data_path ./data   \
--out_path ./temp_dygie_output.json \
--cuda_device <optional id>

```

The required pre-trained weights, specifically the .tar.gz archive files, are available for download from the following website [PhysioNet](https://physionet.org/content/radgraph/1.0.0/).

## 
