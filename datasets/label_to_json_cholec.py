import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import math
import random

# All_phase
'''
{
    'item': {
        'frames': 54930,
        'original_flow': [0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        'compress_flow': [0, 1, 3, 1, 2, 4, 5, 6]
        'n_tuple': [[-1, 0, 1], [0, 1, 3], [1, 3, 1], [3, 1, 2], [1, 2, 4], [2, 4, 5], [4, 5, 6], [5, 6, 7]]
    }
}
'''

# target
'''
sample = {
    'item_path': item_path,
    'last': last_status,
    'current': current_status,
    'next': next_status,
    'segment':[start,end] frame id
    'n_frames': n_frames
    'frames': [111,112,113,114,115,...,198]
    'subset': validation, test, train (one type)
}
'''


def format_arr_folder(arr_str):
    return '_'.join(str(e) for e in arr_str)


def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.iloc[i, 1])
    return labels


def get_frames(original_flow):
    start_pos = 0
    frame_infos = []
    for index in range(len(original_flow)):
        if original_flow[index - 1] != original_flow[index]:
            sample = {
                'segment': [start_pos, index],
                'n_frames': index + 1 - start_pos,
                'frames': np.arange(start_pos, index + 1).tolist()
            }
            frame_infos.append(sample)
            start_pos = index + 1
        if index == len(original_flow) - 1:
            sample = {
                'segment': [start_pos, index + 1],
                'n_frames': index + 2 - start_pos,
                'frames': np.arange(start_pos, index + 2).tolist()
            }
            frame_infos.append(sample)
    return frame_infos[1:]


def reset_subset(array_length):
    subset = []
    train_size = math.floor(array_length * 0.6)
    test_size = math.floor(array_length * 0.3)
    val_size = array_length - train_size - test_size
    for index in range(train_size):
        subset.append('train')
    for index in range(test_size):
        subset.append('test')
    for index in range(val_size):
        subset.append('val')
    random.shuffle(subset)
    return subset


def get_finetune(slices):
    slice_length = len(slices)
    train_size = math.floor(slice_length * 0.6)
    test_size = math.floor(slice_length * 0.3)
    val_size = slice_length - train_size - test_size
    train_slice = random.sample(slices, train_size)
    test_slice = random.sample(slices, test_size)
    val_slice = random.sample(slices, val_size)
    return train_slice, test_slice, val_slice


def make_json_file(label_path, txt_label_path, file_name):
    label_json = {}
    with open(label_path, 'rb') as file:
        label_data = pickle.load(file)
    class_name = load_labels(txt_label_path)
    label_json['labels'] = class_name
    label_json['database'] = []
    for item in label_data.keys():
        print('Processing {}'.format(item))
        label_info = label_data[item]
        original_flow = label_info['original_flow']
        label_frame = get_frames(original_flow)
        index = 0
        for tuple_item in label_info['n_tuple']:
            current_item = label_frame[index]
            train_slice, test_slice, val_slice = get_finetune(current_item['frames'])
            sample = {
                'video': item,
                'step': format_arr_folder(tuple_item),
                'segment': current_item['segment'],
                'n_frames': current_item['n_frames'],
                'last': class_name[tuple_item[0] + 1],
                'current': class_name[tuple_item[1] + 1],
                'next': class_name[tuple_item[2] + 1],
                'frame_indices': current_item['frames'],
                'train': train_slice,
                'test': test_slice,
                'val': val_slice
            }
            label_json['database'].append(sample)
            index = index + 1
    subset = reset_subset(len(label_json['database']))
    for index in range(len(subset)):
        label_json['database'][index]['subset'] = subset[index]
    with open(os.path.join(os.path.split(txt_label_path)[0], file_name + ".json"), 'w') as file:
        json.dump(label_json, file)
    with open(os.path.join(os.path.split(txt_label_path)[0], file_name + ".pkl"), 'wb') as file:
        pickle.dump(label_json, file)


if __name__ == '__main__':
    label = sys.argv[1]
    txt_label = sys.argv[2]
    set_name = sys.argv[3]
    make_json_file(label, txt_label, set_name)
