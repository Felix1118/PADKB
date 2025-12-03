import numpy as np
from einops import rearrange
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets.transforms import DataTransforms
from constants import *
from datasets.data_module import DataModule
from datasets.pretrain_dataset import (MultimodalPretrainingDataset, 
                                            multimodal_collate_fn)
from models.mgca.mgca_module import MGCA
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# Load pretrained model
ckpt_path = "./pretrained/vit_base.ckpt"
model = MGCA.load_from_checkpoint(ckpt_path, strict=False)
save_list=[]
# define datamodule
datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                        DataTransforms, 1, 1, 0) 

for index, batch in enumerate(tqdm(datamodule.val_dataloader(),desc='Have been Generated')):
    with torch.no_grad():
        # print(batch['path'][0])
        # print(batch['entity_position_index'])
        _, patch_feat_q = model.img_encoder_q(batch["imgs"])
        patch_emb_q = model.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        report_feat_q, word_feat_q, word_attn_q, sents = model.text_encoder_q(batch["caption_ids"].unsqueeze(0), batch["attention_mask"].unsqueeze(0), batch["token_type_ids"].unsqueeze(0))
        word_emb_q = model.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, dim=-1)
        bz = patch_feat_q.size(0)
        _, word_atten= model.word_local_atten_layer(word_emb_q, patch_emb_q, patch_emb_q)
        # print(batch['entity_position_index'])
        RadGraph_entity_atten=word_atten[:,batch['entity_position_index'],:]
      
        # print(" ".join([x for x in sents[0] if x != "[PAD]"]))
        # sum_last_dim = RadGraph_entity_atten.sum(dim=2)
        # print(sum_last_dim)
        img = (batch["imgs"].cpu().numpy() * 0.5 + 0.5).clip(0, 1)[0].transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        overlay=img.copy()
        attention_map=RadGraph_entity_atten.sum(dim=1)
        Interval_length=attention_map.max()-attention_map.min()
        attention_map=(attention_map-attention_map.min())/Interval_length
        atten_map = rearrange(attention_map, "b (p1 p2) -> b p1 p2", p1=14, p2=14).squeeze()
        pathology_aware_prior=atten_map.cpu().numpy().astype(np.float16)
       
        save_list.append({batch['path'][0]:pathology_aware_prior*5})
        # if index%1000==0:
        #    print(batch['entity_position_index'])
        #    print(" ".join([x for x in sents[0] if x != "[PAD]"]))
        #    atten_map = ((atten_map.detach().cpu().numpy())*255).astype(np.uint8)
        #    atten_map = cv2.resize(atten_map, (224, 224), interpolation=cv2.INTER_CUBIC)
        #    attention_colormap = cv2.applyColorMap(atten_map, cv2.COLORMAP_JET)
        #    for i in range(14):
        #         for j in range(14):
        #           top_left_x = j * 16
        #           top_left_y = i * 16
        #           bottom_right_x = (j + 1) *16
        #           bottom_right_y = (i + 1) *16
        #           color = attention_colormap[top_left_y, top_left_x].tolist()
        #           cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, -1)
        #    img_with_heatmap = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        #    img_with_heatmap = cv2.cvtColor(img_with_heatmap, cv2.COLOR_BGR2RGB)
        #    fig = plt.figure(figsize=(12, 6))  
        #    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1]) 
        #    ax1 = fig.add_subplot(gs[0, 0])  
        #    ax2 = fig.add_subplot(gs[0, 1])  
        #    ax1.imshow(img)
        #    ax1.set_title("Original Image", fontsize=12)
        #    ax1.axis('off')
        #    ax2.imshow(img_with_heatmap)
        #    ax2.set_title("Image with Attention Heatmap", fontsize=12)
        #    ax2.axis('off')
        #    ax_text = fig.add_subplot(gs[1, :])  # 占用全部列
        #    ax_text.text(0.5, 0.5, 
        #      "Visualization Example", 
        #      ha='center', va='center', fontsize=14)
        #    ax_text.axis('off')
        #    plt.tight_layout()
        #    plt.show()
result_np=np.asarray(save_list)
np.save('pathology_aware_prior.npy',result_np)
print('finished')