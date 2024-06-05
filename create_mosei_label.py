import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = '/root/autodl-tmp/datasets/cmu_mosei'
mosei_emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear']

if __name__=='__main__':

    mosei_label_df = pd.read_csv('exp_mmsa/data_preprocessing/mosei_label.csv')
    original_labels = h5py.File(os.path.join(DATA_PATH,'CMU_MOSEI_Labels.csd'))

    # video_ids
    video_ids = mosei_label_df['video_id'].unique().tolist()
    for video_id in tqdm(video_ids, desc='Processing items', unit='item'):

        # video original labels
        original_features = original_labels['All Labels']['data'][video_id]['features'][()]
        original_intervals = original_labels['All Labels']['data'][video_id]['intervals'][()]

        # sort intervals
        sorted_intervals = np.argsort(original_intervals[:, 1])
        sorted_features = original_features[sorted_intervals]
        emotion_labels = sorted_features[:, 1:].T

        # clip_id
        cur_video_df = mosei_label_df[mosei_label_df['video_id'] == video_id]
        rank = cur_video_df['clip_id'].rank(method='min', ascending=True).astype(int) - 1
        rank = rank.tolist()
        # cur_video_df = cur_video_df.sort_values(by='clip_id')

        if len(cur_video_df) != emotion_labels.shape[1]:
            print(video_id)
            # emotion_labels = emotion_labels[:, :len(clip_indices)]
            continue # need manmual operation
        
        emotion_labels = emotion_labels[:, rank]
        emotion_labels = {k: v for k, v in zip(mosei_emotions, emotion_labels)}
        # assign emotion labels
        for i, emotion in enumerate(mosei_emotions):
            mosei_label_df.loc[mosei_label_df['video_id'] == video_id, emotion] = emotion_labels[emotion]

    mosei_label_df.to_csv('exp_mmsa/data_preprocessing/mosei_all_label.csv')
    print("------------The mosei dataset label file is complete------------")

