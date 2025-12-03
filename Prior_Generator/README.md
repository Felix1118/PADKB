#  Patch Attention Map Generation Pipeline

## Foundational Environment Setup

The patch attention map generation relies on the code and pre-trained weights of the MGCA model. This process requires setting up an environment consistent with the MGCA model specifications, for which detailed instructions can be found in the [official MGCA implementation](https://github.com/HKU-MedAI/MGCA). The corresponding [pre-trained model weights](https://drive.google.com/drive/folders/15_mP9Lqq2H15R53qlKn3l_xzGVzi9jX9) can also be obtained from the official MGCA repository.


## Generate Patch Attention Map

Execute [visualize_attention_map.py]() to load the official MGCA pre-trained weights and compute the Prior Patch Attention Map based on the extracted medical entity terms.
