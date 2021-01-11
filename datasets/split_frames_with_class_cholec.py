import os
import pickle
import shutil
import sys
from tqdm import tqdm

def format_arr_folder(arr_str):

    return '_'.join(str(e) for e in arr_str)


def split_frames_with_class(frame_path, label_path):
    """split path into folder

        Args:
            frame_path (str): frame path e.g:  data/Frames/Full
            label_path (str): label path e.g:  data/Annotation/Phase/all_phase.pkl
    """
    with open(label_path, 'rb') as file:
        label_data = pickle.load(file)
    for item in label_data.keys():
        # load data
        dst_path = os.path.join(frame_path, item)
        flow_data = label_data[item]
        frames = flow_data["frames"]
        original_flow = flow_data["original_flow"]
        n_tuple = flow_data["n_tuple"]
        # make folder
        for folder_name in n_tuple:
            print('Make folder {}'.format(format_arr_folder(folder_name)), end="\r")
            try:
                os.mkdir(os.path.join(dst_path, format_arr_folder(folder_name)))
            except FileExistsError:
                print('Folder {} exist'.format(format_arr_folder(folder_name)), end="\r")

        # start move file
        dst_folder_index = 0
        pbar = tqdm(range(0, frames))
        for index in pbar:
            if index > 1:
                if original_flow[index] != original_flow[index - 1]:
                    dst_folder_index = dst_folder_index + 1
            folder_name = format_arr_folder(n_tuple[dst_folder_index])
            current_dst = os.path.join(dst_path, folder_name)
            current_file = os.path.join(dst_path, 'image_%08d.jpg' % (index + 1))
            # unsafely move
            try:
                shutil.copy(current_file, current_dst)
                # os.unlink(current_file)
                pbar.set_description('Copy {} to {}'.format(current_file, current_dst))
            except FileNotFoundError:
                continue


if __name__ == '__main__':
    path_of_frame = sys.argv[1]
    path_of_label = sys.argv[2]
    split_frames_with_class(path_of_frame, path_of_label)
