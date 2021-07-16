import pickle

import numpy as np
import torch

from MTL_hparam import dataset_path


class Rap_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_data):
        self.encodings = input_data
        self.durations = input_data['durations']
        self.labels = input_data['labels']
        self.pitches = input_data['pitches']

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['durations'] = torch.tensor(self.durations[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        item['pitches'] = torch.tensor(self.pitches[idx])
        return item

    def __len__(self):
        return len(self.durations)


def grab_train_data():
    raw_data = []
    # loading training data
    for item in dataset_path:
        with open(item, "rb") as f:
            data = pickle.load(f)
            raw_data.extend(data)

    #raw_data = raw_data[0:1000]
    # preparing training data
    tokens = []
    durations = []
    beats = []
    pitches = []

    for line in raw_data:
        line['words'] = list(line['words'])
        line['durs'] = list(map(float, line['durs']))
        assert len(line['words']) == len(line['durs'])
        assert len(line['words']) == len(line['beats'])
        assert len(line['words']) == len(line['pitches'])
        while '' in line['words']:
            idx = line['words'].index('')
            del line['words'][idx]
            del line['durs'][idx]
            del line['beats'][idx]
            del line['pitches'][idx]

        # logarithm
        line['durs'] = np.log(np.array(line['durs'])).tolist()

        # pitch_shift
        line['pitches'] = (np.array(line['pitches']) - 36).tolist()
        if np.min(np.array(line['pitches'])) < 0:
            pitches_tmp = []
            for item in line['pitches']:
                item = max(0, item)
                pitches_tmp.append(item)
            line['pitches'] = pitches_tmp
        assert np.min(np.array(line['pitches'])) >= 0
        assert np.max(np.array(line['pitches'])) <= 47

        # length check
        assert len(line['words']) == len(line['durs'])
        assert len(line['words']) == len(line['beats'])
        assert len(line['words']) == len(line['pitches'])
        tokens.append(line['words'])
        durations.append(line['durs'])
        beats.append(line['beats'])
        pitches.append(line['pitches'])

    training_data = {'tokens': tokens, 'beats': beats, 'durations': durations, 'pitches': pitches}
    return training_data
