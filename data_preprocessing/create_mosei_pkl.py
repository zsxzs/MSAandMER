import pickle
import numpy as np
import pandas as pd

label_csv = '/root/autodl-tmp/exp_mmsa/data_preprocessing/mosei_all_label.csv'
aligned_pkl_path = '/root/autodl-tmp/MMSA/MOSEI/Processed/aligned_50.pkl'
unaligned_pkl_path = '/root/autodl-tmp/MMSA/MOSEI/Processed/unaligned_50.pkl'
new_aligned_pkl_path = '/root/autodl-tmp/exp_mmsa/data_preprocessing/aligned_50.pkl'
new_unaligned_pkl_path = '/root/autodl-tmp/exp_mmsa/data_preprocessing/unaligned_50.pkl'

if __name__=='__main__':

    all_label_df = pd.read_csv(label_csv)
    emotions = ['happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear']

    train_label_df = all_label_df[all_label_df['mode'] == 'train']
    valid_label_df = all_label_df[all_label_df['mode'] == 'valid']
    test_label_df = all_label_df[all_label_df['mode'] == 'test']

    train_emotion_labels = train_label_df[emotions].to_numpy()
    valid_emotion_labels = valid_label_df[emotions].to_numpy()
    test_emotion_labels = test_label_df[emotions].to_numpy()

    # aligned
    with open(aligned_pkl_path, 'rb') as f:
        aligned_data = pickle.load(f)
    
    aligned_data['train']['emotion_labels'] = train_emotion_labels
    aligned_data['valid']['emotion_labels'] = valid_emotion_labels
    aligned_data['test']['emotion_labels'] = test_emotion_labels

    with open(new_aligned_pkl_path, 'wb') as f:
        pickle.dump(aligned_data, f)

    # unaligned 
    with open(unaligned_pkl_path, 'rb') as f:
        unaligned_data = pickle.load(f)

    unaligned_data['train']['emotion_labels'] = train_emotion_labels
    unaligned_data['valid']['emotion_labels'] = valid_emotion_labels
    unaligned_data['test']['emotion_labels'] = test_emotion_labels

    with open(new_unaligned_pkl_path, 'wb') as f:
        pickle.dump(unaligned_data, f)