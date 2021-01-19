import torch.utils.data as data
import torch
from PIL import Image
import os
import math
import functools
import pickle
import json
import copy
from tqdm import tqdm
import random
from datasets.label_to_json_cholec import reset_subset
import logging


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    """accimage loader

    Args:
        path:
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    """
    Args:
        video_dir_path:
        frame_indices:
        image_loader:

    Returns:
        video: video indices
    """
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:08d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            logging.warning(image_path)
            logging.error("Error: File not exits!")
            return video
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    """
    Args:
        data_file_path:
    """
    with open(data_file_path, 'rb') as data_file:
        return pickle.load(data_file)


def get_class_labels(data):
    """
    Args:
        data:
    """
    class_labels_map = {}
    index = -1
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_subset(data, subset):
    """
    Args:
        data:
        subset:
    """
    subset_annotations = []
    for item in data['database']:
        if item['subset'] == subset:
            subset_annotations.append(item)
        if subset == "all":
            subset_annotations.append(item)
    return subset_annotations


def get_related_sample(current_sample, datalist, is_previous):
    """
    Get related sample
    :param current_sample:
    :param datalist:
    :param is_previous:
    :return:
    """
    current_step = current_sample['step'].split('_')
    for item in datalist:
        to_find_item_step = item['step'].split('_')
        if is_previous:
            if (to_find_item_step[1] == current_step[0]) & (to_find_item_step[2] == current_step[1]):
                return item
        else:
            if (to_find_item_step[0] == current_step[1]) & (to_find_item_step[1] == current_step[2]):
                return item


def make_finetune_dataset(root_dir, annotation_path, subset):
    data = load_annotation_data(annotation_path)
    database = data['database']
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    dataset = []
    pbar = tqdm(range(len(database)))
    for i in pbar:
        item = database[i]
        img_slice = item[subset]
        for img_index in img_slice:
            sample = {
                'path': os.path.join(root_dir, item['video'], item['step'], 'image_{:08d}.jpg'.format(img_index)),
                'last': class_to_idx[item['last']], 'current': class_to_idx[item['current']],
                'next': class_to_idx[item['next']]}
            dataset.append(sample)
    random.shuffle(dataset)
    return dataset, idx_to_class


def make_pretrain_dataset(root_dir, annotation_path, subset, n_sample_for_each_video, sample_duration):
    """Formatted as: sample = {
        'video' : video path ', segment' : start and end frame ', n_frames' : total frames, 'video_id' ：
        video id , 'label' : true label , 'frame_indices' : frames id

    }

    Args:
        root_dir:
        annotation_path:
        subset:
        n_sample_for_each_video:
        sample_duration:
    """
    data = load_annotation_data(annotation_path)
    sub_data = get_subset(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    pbar = tqdm(range(len(sub_data)))
    for i in pbar:
        pbar.set_description('loading {} set \t [{}/{}]'.format(subset, i + 1, len(sub_data)))
        sample = sub_data[i]
        video_path = os.path.join(root_dir, sample['video'], sample['step'])
        sample['video_path'] = video_path
        sample['last'] = class_to_idx[sub_data[i]['last']]
        sample['current'] = class_to_idx[sub_data[i]['current']]
        sample['next'] = class_to_idx[sub_data[i]['next']]

        n_frames = sample['n_frames']  # n_frames handle the step of the video
        start_frame = sample['segment'][0]
        end_frame = sample['segment'][1]
        if n_sample_for_each_video == 1:  # Add all image for validate each activity,
            sample['frame_indices'] = list(range(start_frame, end_frame + 1))
            dataset.append(sample)
        else:  # split video into small samples with step
            if n_sample_for_each_video > 1:
                step = max(1, math.ceil((n_frames - 1 - sample_duration) /
                                        (n_sample_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(start_frame, end_frame, step):
                sample_j = copy.deepcopy(sample)
                if j + sample_duration < n_frames + start_frame:
                    frame_indices = list(range(j, min(n_frames + start_frame, j + sample_duration)))
                else:
                    frame_indices = list(range(n_frames + start_frame - sample_duration, n_frames + start_frame))
                sample_j['frame_indices'] = frame_indices
                dataset.append(sample_j)

    return dataset, idx_to_class


def make_guide_dataset(root_dir, annotation_path, subset, n_sample_for_each_video, sample_duration):
    data = load_annotation_data(annotation_path)
    full_data = get_subset(data, "all")  # add all data
    class_to_idx = get_class_labels(data)
    idx_to_class_all = {}
    for name, label in class_to_idx.items():
        idx_to_class_all[label] = name
    # new added for load and save
    guide_pkl = os.path.join(root_dir, "full.pkl")
    if os.path.exists(guide_pkl):
        with open(guide_pkl, "rb") as file:
            dataset = pickle.load(file)
    else:
        with open(guide_pkl, "wb") as file:
            dataset = []
            pbar = tqdm(range(len(full_data)))
            for i in pbar:
                pbar.set_description('loading guide data \t [{}/{}]'.format(i + 1, len(full_data)))
                sample = full_data[i]
                if sample['next'] != 'End' and sample['next'] != 7 \
                        and sample['last'] != 'Start' and sample['last'] != -1:
                    # if not start and end, will load previous and future
                    pre_sample = get_related_sample(sample, full_data, is_previous=True)
                    future_sample = get_related_sample(sample, full_data, is_previous=False)
                    sample_dit = {'previous_sample': pre_sample,
                                  'current_sample': sample,
                                  'future_sample': future_sample}
                    sample_status_dict = {'previous_sample', 'current_sample', 'future_sample'}
                    guide_item = {}
                    for sample_status in sample_status_dict:
                        sample_item = sample_dit[sample_status]
                        video_path = os.path.join(root_dir, sample_item['video'], sample_item['step'])
                        sample_item['video_path'] = video_path
                        try:
                            sample_item['last'] = class_to_idx[sample_item['last']]
                            sample_item['current'] = class_to_idx[sample_item['current']]
                            sample_item['next'] = class_to_idx[sample_item['next']]
                        except KeyError:
                            pass
                        n_frames = sample_item['n_frames']  # n_frames handle the step of the video
                        start_frame = sample_item['segment'][0]
                        end_frame = sample_item['segment'][1]
                        if n_sample_for_each_video == 1:  # Number of validation samples for each activity, for test
                            sample_item['frame_indices'] = list(range(start_frame, end_frame + 1))
                            guide_item[sample_status] = sample
                        else:  # split video into small samples with step
                            if n_sample_for_each_video > 1:
                                step = max(1, math.ceil((n_frames - 1 - sample_duration) /
                                                        (n_sample_for_each_video - 1)))
                            else:
                                step = sample_duration
                            for j in range(start_frame, end_frame, step):
                                sample_j = copy.deepcopy(sample_item)
                                if j + sample_duration < n_frames + start_frame:
                                    frame_indices = list(
                                        range(j, min(n_frames + start_frame, j + sample_duration)))
                                else:
                                    frame_indices = list(
                                        range(n_frames + start_frame - sample_duration, n_frames + start_frame))
                            sample_j['frame_indices'] = frame_indices
                            guide_item[sample_status] = sample_j
                    dataset.append(guide_item)
            pickle.dump(dataset, file)
    # split guided
    data_length = len(dataset)
    data_dist = reset_subset(data_length)
    for item_index in range(len(data_dist)):
        dataset[item_index]['target'] = dataset[item_index]['current_sample']['step']
        dataset[item_index]['subset'] = data_dist[item_index]
    # split train val test
    sub_dataset = []
    for item in dataset:
        if item['subset'] == subset:
            sub_dataset.append(item)
    return sub_dataset, idx_to_class_all


def make_dataloader_dict(root_dir, annotation_path, is_finetune, is_guide, batch_size, spatial_transform,
                         temporal_transform,
                         target_transform=[],
                         sample_duration=0,
                         n_sample_for_each_video=0):
    train_set = CholecData(root_path=root_dir,
                           annotation_path=annotation_path, target_transform=target_transform,
                           spatial_transform=spatial_transform, temporal_transform=temporal_transform,
                           subset='train', is_finetune=is_finetune, is_guide=is_guide, sample_duration=sample_duration,
                           n_samples_for_each_video=n_sample_for_each_video)
    val_set = CholecData(root_path=root_dir,
                         annotation_path=annotation_path, target_transform=target_transform,
                         spatial_transform=spatial_transform, temporal_transform=temporal_transform,
                         subset='val', is_finetune=is_finetune, is_guide=is_guide, sample_duration=sample_duration,
                         n_samples_for_each_video=n_sample_for_each_video)
    test_set = CholecData(root_path=root_dir,
                          annotation_path=annotation_path, target_transform=target_transform,
                          spatial_transform=spatial_transform, temporal_transform=temporal_transform,
                          subset='test', is_finetune=is_finetune, is_guide=is_guide, sample_duration=sample_duration,
                          n_samples_for_each_video=n_sample_for_each_video)
    image_datasets = {'train': train_set, 'val': val_set, 'test': test_set}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val', 'test']}
    return dataloaders_dict


class CholecData(data.Dataset):
    """A dataset formatted with Cholec 2019"""

    def __init__(self,
                 root_path,
                 annotation_path,  # use pkl file
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 is_finetune=False,
                 is_guide=False):
        """
        Args:
            root_path:
            annotation_path:
            subset:
            n_samples_for_each_video:
            spatial_transform:
            temporal_transform:
            target_transform:
            sample_duration:
            get_loader:
        """
        self.is_finetune = is_finetune
        self.is_guide = is_guide
        if is_finetune:  # finetune
            self.data, self.class_name = make_finetune_dataset(root_path, annotation_path, subset)
        elif is_guide:  # guide
            self.data, self.class_name = make_guide_dataset(root_path, annotation_path, subset,
                                                            n_samples_for_each_video,
                                                            sample_duration)
        elif not is_guide and not is_finetune:  # pretrain
            self.data, self.class_name = make_pretrain_dataset(root_path, annotation_path, subset,
                                                               n_samples_for_each_video,
                                                               sample_duration)
        else:
            logging.error("label set error")
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index:
        """
        if self.is_finetune:  # finetune
            path = self.data[index]['path']
            target = self.data[index]
            img_loader = get_default_image_loader()
            clip = img_loader(path)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = self.spatial_transform(clip)
            if self.target_transform is not None:
                target = self.target_transform(target)
                target = torch.tensor(target)
            target = torch.tensor(target)
            return clip, target
        elif self.is_guide:  # guide
            clip = {}
            status_name = {'previous_sample', 'current_sample', 'future_sample'}
            for status_item in status_name:
                path = self.data[index][status_item]['video_path']
                frame_indices = self.data[index][status_item]['frame_indices']
                if self.temporal_transform is not None:
                    frame_indices = self.temporal_transform(frame_indices)
                clip[status_item] = self.loader(path, frame_indices)
                if self.spatial_transform is not None:
                    self.spatial_transform.randomize_parameters()
                    clip[status_item] = [self.spatial_transform(img) for img in clip[status_item]]
                clip[status_item] = torch.stack(clip[status_item], 0)
                target = self.data[index]['target']
                if self.target_transform is not None:  # why one label
                    target = self.target_transform(target)
                    target = torch.tensor(target)
            return clip, target
        elif not self.is_guide and not self.is_finetune:  # pretrain
            path = self.data[index]['video_path']
            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            clip = self.loader(path, frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0)
            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
                target = torch.tensor(target)
            return clip, target

    def __len__(self):
        return len(self.data)

    def target_category(self, target):
        '''
        Cover one-hot to label
        '''
        if target not in self.class_name.keys():
            raise RuntimeError('Not found label!')
        c_label = self.class_name(target)
        return c_label


if __name__ == "__main__":
    # only for test
    from utils.spatial_transforms import (
        Compose, Scale, ToTensor)
    from utils.target_transforms import (FlowLabel, ClassLabel)
    from utils.mylogger import setup_logger

    setup_logger("debug", "debug", file_dir='test')
    root_path = 'data/Frames/Full'
    anna_path = 'data/Annotation/data.pkl'
    dataset = CholecData(root_path=root_path, n_samples_for_each_video=30, sample_duration=20,
                         annotation_path=anna_path, subset='train')
    logging.info("only debug in cholecdata")
    for i, (inputs, targets) in enumerate(dataset):
        logging.info(targets)
