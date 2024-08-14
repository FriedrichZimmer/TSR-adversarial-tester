"""
Copyright (c) 2024 Friedrich Zimmer
Convert all images in a folder into a video
"""


from argparse import ArgumentParser

from cv2 import imread, IMREAD_COLOR, VideoWriter, VideoWriter_fourcc
import os


FPS = 10


def gen_video(image_folder, fps=10, vformat='mp4'):
    """
    generate video out of all images in a folder and stores it in a _video subfolder

    Args:
        image_folder (str): folder with images
        fps (int): Frames per second
        vformat (str): video format. Can be mp4 or avi
    """

    video_folder = os.path.join(image_folder, '_videos')
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    if vformat == 'mp4':
        export_video = os.path.join(video_folder, "video.mp4")
    else:
        export_video = os.path.join(video_folder, "video.avi")
    print(f'creating video at {export_video}')

    # load all images of folder into an
    frames = []
    for path in os.listdir(image_folder):
        file = os.path.join(image_folder, path)
        if os.path.isfile(file) and file[len(file) - 3:] == "png":
            img = imread(file, IMREAD_COLOR)
            frames.append(img)

    if len(frames) > 0:
        # get image resolution from first image
        h, w, c = frames[0].shape
        video_dim = (w, h)

        if vformat == 'mp4':
            vid_writer = VideoWriter(export_video, VideoWriter_fourcc(*'mp4v'), fps, video_dim)
        else:
            vid_writer = VideoWriter(export_video, VideoWriter_fourcc(*'XVID'), fps, video_dim)

        for frame in frames:
            vid_writer.write(frame)
        vid_writer.release()
        print(f'Created video {export_video}')
    else:
        print('No videos found in this folder')


def gen_all_videos(result_folder, fps, vformat='mp4'):
    """ generate video out of all images in all folders of a test

        Args:
        image_folder (str): folder with images
        fps (int): Frames per second
        vformat (str): video format. Can be mp4 or avi
    """
    for sign in os.listdir(result_folder):
        sign_folder = os.path.join(result_folder, sign)
        if os.path.isfile(sign_folder):  # skip the statistics files stored in the main folder
            continue
        for cam in os.listdir(sign_folder):
            # define the folders used for the results
            current_image_folder = os.path.join(sign_folder, cam)
            gen_video(current_image_folder, fps, vformat)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source', type=str, help='Filepath and name of the results directory')
    parser.add_argument('-fps', type=int, help='Frames per Second', default=10)
    parser.add_argument('-format', type=str, help='Video Format (mp4 or avi)', default='mp4')
    args = parser.parse_args()
    gen_video(args.source, args.fps, args.format)
