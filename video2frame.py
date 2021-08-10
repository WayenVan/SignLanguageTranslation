import typing

import numpy as np
import pickle
import cv2
import yaml


def split_video_data(source_path, config_path, target_dir, sample_num=8):

    #load information from yaml
    with open(config_path) as config:
        config_dict = yaml.safe_load(config)

    id = config_dict["id"]
    sentence = config_dict["sentence"]
    signs = config_dict["signs"]

    #open video
    video = cv2.VideoCapture(source_path)

    for sign in signs:
        sign_gloss = sign["gloss"]
        sign_begin = sign["begin"]
        sign_end = sign["end"]

        sample_frame = np.linspace(sign_begin, sign_end, num=sample_num, endpoint=True)
        sample_frame = np.around(sample_frame)

        for frame_index in sample_frame:

            #read frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            result, frame = video.read()
            assert result==True

            cv2.imwrite(target_dir+'/'+"sign"+"_"+sign_gloss+"_"+str(int(frame_index))+".jpg", frame)
