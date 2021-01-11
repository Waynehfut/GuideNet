from datasets.cholecdata import CholecData
import os
def get_fine_tune_train_set(opt,spatial_transform,target_transform):
    assert opt.datasets in ['cholec2019', 'cholec80', 'miccai2016']
    if opt.datasets == 'cholec2019':
        training_data = CholecData(
            opt.video_path,
            opt.annotation_path,
            'train',
            spatial_transform=spatial_transform, # image transform
            temporal_transform=None,
            target_transform=target_transform, # label transform
            # TODO remove next line, only for test
            n_samples_for_each_video=100)
    else:
        print('if you want add new set, please add as datasets.ch19_datasets')

    return training_data

def get_fine_tune_val_set(opt,spatial_transform,target_transform):
    assert opt.datasets in ['cholec2019', 'cholec80', 'miccai2016']
    if opt.datasets == 'cholec2019':
        training_data = CholecData(
            opt.video_path,
            opt.annotation_path,
            'val',
            spatial_transform=spatial_transform,
            temporal_transform=None,
            target_transform=target_transform,
            # TODO remove next line, only for test
            n_samples_for_each_video=100)
    else:
        print('if you want add new set, please add as datasets.ch19_datasets')

    return training_data

def get_fine_tune_test_set(opt,data_transforms):
    assert opt.datasets in ['cholec2019', 'cholec80', 'miccai2016']
    if opt.datasets == 'cholec2019':
        training_data = CholecData(
            opt.video_path,
            opt.annotation_path,
            'test',
            spatial_transform=data_transforms,
            temporal_transform=None,
            target_transform=None,
            # TODO remove next line, only for test
            n_samples_for_each_video=100)
    else:
        print('if you want add new set, please add as datasets.ch19_datasets')

    return training_data

def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.datasets in ['cholec2019', 'cholec80', 'miccai2016']
    if opt.datasets == 'cholec2019':
        training_data = CholecData(
            opt.video_path,
            opt.annotation_path,
            'train',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            # TODO remove next line, only for test
            n_samples_for_each_video=100)
    else:
        print('if you want add new set, please add as datasets.ch19_datasets')

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.datasets in ['cholec2019', 'cholec80', 'miccai2016']
    if opt.datasets == 'cholec2019':
        validation_data = CholecData(
            opt.video_path,
            opt.annotation_path,
            'val',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform
        )
    else:
        print('if you want add new set, please add as datasets.ch19_datasets')

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.datasets in ['cholec2019', 'cholec80', 'miccai2016']
    if opt.datasets == 'cholec2019':
        test_data = CholecData(
            opt.video_path,
            opt.annotation_path,
            'test',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform
        )
    else:
        print('if you want add new set, please add as datasets.ch19_datasets')

    return test_data

