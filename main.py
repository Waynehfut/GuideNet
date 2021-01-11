is_test=True

import os
if is_test:
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9"  # identify the GPU here.
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # identify the GPU here.

from torch import nn
import torch
from torch import optim
from datasets.cholecdata import make_dataloader_dict
from train_val import train_finetune, pretrain_guide, train_guide, model_test, model_guide_test
from utils.mylogger import setup_logger
import logging
import utils.spatial_transforms as st
import utils.target_transforms as tt
import utils.temporal_transforms as tempt

sample_duration = 16
n_sample_for_each_video = 32
batch_size = 256

# Stage
is_finetune = True
is_pretrain_guide = True
is_guide = True

# train
is_finetune_train = True
is_pretrain_train = True
is_guide_train = True

# test
is_finetune_test = True
is_pretrain_test = True
is_guide_test = True

model_name = ""
log_type = ""
if is_finetune:
    log_type = "train_val"
    model_name = "finetune"
    if is_finetune_test and not is_finetune_train:
        log_type = "test_only"
elif is_pretrain_guide:
    log_type = "train_val"
    model_name = "pretrain_guide"
    if is_pretrain_test and not is_pretrain_train:
        log_type = "test_only"
elif is_guide:
    log_type = "train_val"
    model_name = "guide"
    if is_guide_test and not is_guide_train:
        log_type = "test_only"

setup_logger(model_name, log_type)

logging.info('Let us using {} GPUs'.format(torch.cuda.device_count()))
root_dir = 'data/Frames/Full'
anna_dir = 'data/Annotation/data.pkl'
model_name = 'resnet50'
finetune_weight_out = 'out/checkpoint/' + model_name + '_finetune_{}.pkl'
pre_guide_weight_out = 'out/checkpoint/cnnlstm_preguide_{}.pkl'
guided_weight_out = 'out/checkpoint/guided_cnnlstm_preguide_{}.pkl'

spatial_transform = st.Compose([st.Scale((224, 224)), st.ToTensor()])

temporal_transform = [tempt.TemporalSubsampling(2), tempt.TemporalRandomCrop(2), tempt.TemporalCenterCrop(16)]
temporal_transform = tempt.Compose(temporal_transform)

if is_finetune:  # if finetune the CNN model
    is_pretrain_guide = False
    is_guide = False
    dataloaders_dict = make_dataloader_dict(root_dir=root_dir, annotation_path=anna_dir, is_finetune=is_finetune,
                                            is_guide=is_guide, batch_size=batch_size,
                                            spatial_transform=spatial_transform, temporal_transform=None)
    if is_finetune_train:
        model_ft, hist = train_finetune(dataloaders_dict, finetune_weight_out, model_name, num_epochs=300)
    if is_finetune_test:  # finetune model test
        model = torch.load(finetune_weight_out.format('best'))
        criterion = nn.CrossEntropyLoss()
        model_test(model, dataloaders_dict, criterion)

elif is_pretrain_guide:  # if pretrain the guide model
    is_finetune = False  # is_finetune must false
    is_guide = False
    target_transform = tt.ClassLabel()
    dataloaders_dict = make_dataloader_dict(root_dir=root_dir, annotation_path=anna_dir, is_finetune=is_finetune,
                                            is_guide=is_guide, batch_size=batch_size,
                                            target_transform=target_transform,
                                            spatial_transform=spatial_transform,
                                            temporal_transform=temporal_transform,
                                            sample_duration=sample_duration,
                                            n_sample_for_each_video=n_sample_for_each_video)
    if is_pretrain_train:
        model_ft, hist = pretrain_guide(dataloaders_dict, finetune_weight_out.format('best'), pre_guide_weight_out,
                                        num_epochs=200)
    if is_pretrain_test:  # finetune model test
        model = torch.load(pre_guide_weight_out.format('best'))
        criterion = nn.CrossEntropyLoss()
        model_test(model, dataloaders_dict, criterion)


elif is_guide:  # if guide model train
    is_finetune = False  # is_finetune must false
    is_pretrain_guide = False
    target_transform = tt.FlowLabel()
    dataloaders_dict = make_dataloader_dict(root_dir=root_dir, annotation_path=anna_dir, is_finetune=is_finetune,
                                            is_guide=is_guide, batch_size=batch_size,
                                            target_transform=target_transform,
                                            spatial_transform=spatial_transform,
                                            temporal_transform=temporal_transform,
                                            sample_duration=sample_duration,
                                            n_sample_for_each_video=n_sample_for_each_video)
    if is_guide_train:
        train_guide(dataloaders_dict, pre_guide_weight_out.format('10'), guided_weight_out, num_epochs=200)
    if is_guide_test:  # finetune model test
        model = torch.load(guided_weight_out.format('best'))
        criterion_pg = nn.CrossEntropyLoss()
        criterion_fu = nn.SmoothL1Loss()
        model_guide_test(model, dataloaders_dict, criterion_pg, criterion_fu)
