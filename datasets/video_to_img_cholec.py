import os
import sys
import subprocess
import traceback


def video_img(dir_path, dst_img_path):
    """Cover video to image frame with extension name '.JPG', this is function
    used the ffmpeg.

    Args:
        dir_path (string): video path input
        dst_img_path (string): image path input
    """

    if not os.path.exists(dst_img_path):
        os.mkdir(dst_img_path)

    for file_name in os.listdir(dir_path):
        if "mp4" not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_img_path, name)
        video_file_path = os.path.join(dir_path, file_name)

        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                    subprocess.call(
                        'rd /s /q  \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.mkdir(dst_directory_path)
                else:
                    continue
            else:
                os.mkdir(dst_directory_path)
        except Exception:
            print(traceback.print_exc())
            continue
        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:540 \"{}/{}/image_%08d.jpg\"'.format(
            video_file_path, dst_img_path, name)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')


if __name__ == "__main__":

    dir_path = sys.argv[1]
    dst_img_path = sys.argv[2]
    video_img(dir_path, dst_img_path)
