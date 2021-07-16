import torch
import numpy as np
from transformers import AdamW
from tqdm import tqdm

from MTL_hparam import batch_size, kwargs, record_file, label_list
from datasets import load_metric

# loading metric
from sklearn.metrics import mean_squared_error

metric = load_metric('./metrics/seqeval.py')


class Trainer:
    def __init__(self, backbone, beat_head, duration_head, pitch_head, beat_criterion=torch.nn.CrossEntropyLoss(),
                 duration_criterion=torch.nn.MSELoss(), pitch_criterion=torch.nn.CrossEntropyLoss(),
                 device=torch.device('cuda')):
        self.device = device
        self.backbone = backbone
        self.beat_head = beat_head
        self.duration_head = duration_head
        self.pitch_head = pitch_head
        self.backbone.fc = torch.nn.Sequential()
        self.beat_model = torch.nn.Sequential(self.backbone, self.beat_head).to(self.device)
        self.duration_model = torch.nn.Sequential(self.backbone, self.duration_head).to(self.device)
        self.pitch_model = torch.nn.Sequential(self.backbone, self.pitch_head).to(self.device)

        if hasattr(self.beat_model, 'module'):
            self.beat_optimizer = AdamW([{'params': self.beat_model.module[0].parameters(), 'lr': 2e-5},
                                         {'params': self.beat_model.module[1].parameters(), 'lr': 1e-4}],
                                        weight_decay=0.01)
        else:
            self.beat_optimizer = AdamW([{'params': self.beat_model[0].parameters(), 'lr': 2e-5},
                                         {'params': self.beat_model[1].parameters(), 'lr': 1e-4}],
                                        weight_decay=0.01)

        if hasattr(self.duration_model, 'module'):
            self.duration_optimizer = AdamW(
                [{'params': self.duration_model.module[0].parameters(), 'lr': 2e-5},
                 {'params': self.duration_model.module[1].parameters(), 'lr': 1e-4}
                 ],
                weight_decay=0.01)
        else:
            self.duration_optimizer = AdamW(
                [{'params': self.duration_model[0].parameters(), 'lr': 2e-5},
                 {'params': self.duration_model[1].parameters(), 'lr': 1e-4}
                 ],
                weight_decay=0.01)

        if hasattr(self.pitch_model, 'module'):
            self.pitch_optimizer = AdamW(
                [{'params': self.pitch_model.module[0].parameters(), 'lr': 2e-5},
                 {'params': self.pitch_model.module[1].parameters(), 'lr': 1e-4}
                 ],
                weight_decay=0.01)
        else:
            self.pitch_optimizer = AdamW(
                [{'params': self.pitch_model[0].parameters(), 'lr': 2e-5},
                 {'params': self.pitch_model[1].parameters(), 'lr': 1e-4}
                 ],
                weight_decay=0.01)

        self.beat_criterion = beat_criterion.to(self.device)
        self.duration_criterion = duration_criterion.to(self.device)
        self.pitch_criterion = pitch_criterion.to(self.device)

        with open(record_file, 'a') as f:
            f.write('lr: \r\n')
            f.write('beat \r\n')
            f.write('backbone: {}\r\n'.format(self.beat_optimizer.state_dict()['param_groups'][0]['lr']))
            f.write('head: {}\r\n'.format(self.beat_optimizer.state_dict()['param_groups'][1]['lr']))
            f.write('duration')
            f.write('backbone: {}\r\n'.format(self.beat_optimizer.state_dict()['param_groups'][0]['lr']))
            f.write('head {}\r\n'.format(self.duration_optimizer.state_dict()['param_groups'][1]['lr']))
            f.write('pitch')
            f.write('backbone: {}\r\n'.format(self.pitch_optimizer.state_dict()['param_groups'][0]['lr']))
            f.write('head {}\r\n'.format(self.pitch_optimizer.state_dict()['param_groups'][1]['lr']))

        print(self.device)

    def train(self, training_set, num_epochs, validation_set=None):
        self.backbone.train()
        self.beat_model.train()
        self.duration_model.train()
        self.pitch_model.train()
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_set)

        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=batch_size, sampler=train_sampler, shuffle=False, **kwargs)
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            labels_pred = []
            labels_true = []
            durs_pred = []
            durs_true = []
            pitches_pred = []
            pitches_true = []
            for item in tqdm(train_loader, desc='Training'):
                # for i, item in enumerate(train_loader):
                input_ids = item['input_ids'].numpy().tolist()
                input_ids = torch.tensor(input_ids).to(self.device)
                attention_mask = item['attention_mask'].numpy().tolist()
                attention_mask = torch.tensor(attention_mask).to(self.device)

                # duration
                dur = item['durations'].numpy().tolist()
                dur = torch.tensor(dur).to(self.device)
                self.duration_optimizer.zero_grad()
                outputs = self.duration_model([input_ids, attention_mask])
                outputs = torch.squeeze(outputs, 1)
                loss = self.duration_criterion(outputs, dur)
                loss.backward()
                self.duration_optimizer.step()
                tmp1 = []
                for ite in outputs.cpu().detach().numpy().tolist():
                    tmp1.extend(ite)
                durs_pred.append(tmp1)
                tmp = []
                for ite in item['durations'].cpu().detach().numpy().tolist():
                    tmp.extend(ite)
                durs_true.append(tmp)

                # pitch
                pitch = item['pitches'].numpy().tolist()
                pitch = torch.tensor(pitch).to(self.device)
                self.pitch_optimizer.zero_grad()
                outputs = self.pitch_model([input_ids, attention_mask])
                outputs = torch.squeeze(outputs, 1)
                trans_outputs = torch.transpose(outputs, 1, 2)
                loss = self.pitch_criterion(trans_outputs, pitch)
                loss.backward()
                self.pitch_optimizer.step()
                tmp1 = []
                for ite in np.argmax(outputs.cpu().detach().numpy().tolist(), axis=-1):
                    tmp1.extend(ite)
                pitches_pred.append(tmp1)
                tmp = []
                for ite in item['pitches'].cpu().detach().numpy().tolist():
                    tmp.extend(ite)
                pitches_true.append(tmp)

                # beat
                label = item['labels'].numpy().tolist()
                label = torch.tensor(label).to(self.device)
                self.beat_optimizer.zero_grad()
                outputs = self.beat_model([input_ids, attention_mask])
                outputs = torch.squeeze(outputs, 1)
                trans_outputs = torch.transpose(outputs, 1, 2)
                loss = self.beat_criterion(trans_outputs, label)
                loss.backward()
                self.beat_optimizer.step()
                tmp1 = []
                for ite in np.argmax(outputs.cpu().detach().numpy().tolist(), axis=-1):
                    tmp1.extend(ite)
                labels_pred.append(tmp1)
                tmp = []
                for ite in item['labels'].cpu().detach().numpy().tolist():
                    tmp.extend(ite)
                labels_true.append(tmp)
                # labels.append(label.cpu().numpy().tolist())
                # Remove ignored index (special tokens)

            true_durations_predictions = [
                [p for (p, d) in zip(prediction, duration) if d != 0]
                for prediction, duration in zip(durs_pred, durs_true)
            ]
            true_durations = [
                [d for (p, d) in zip(prediction, duration) if d != 0]
                for prediction, duration in zip(durs_pred, durs_true)
            ]

            loss_dur = 0
            for i in range(len(true_durations)):
                loss_dur += mean_squared_error(true_durations[i], true_durations_predictions[i])
            loss_dur /= len(true_durations)

            true_labels_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(labels_pred, labels_true)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(labels_pred, labels_true)
            ]
            beat_results = metric.compute(predictions=true_labels_predictions, references=true_labels, suffix=True)

            true_pitches_predictions = [
                ['L{}-P'.format(p) for (p, l) in zip(prediction, pitch) if l != -100]
                for prediction, pitch in zip(pitches_pred, pitches_true)
            ]
            true_pitches = [
                ['L{}-P'.format(l) for (p, l) in zip(prediction, pitch) if l != -100]
                for prediction, pitch in zip(pitches_pred, pitches_true)
            ]
            pitch_results = metric.compute(predictions=true_pitches_predictions, references=true_pitches, suffix=True)

            with open(record_file, 'a') as f:
                f.write('\r\n' + 'epoch: ' + str(epoch) + '\r\n')
                f.write('train: ' + '\r\n')
                f.write('beat score: ' + '\r\n' + str(beat_results) + '\r\n')
                f.write('pitch score: ' + '\r\n' + str(pitch_results) + '\r\n')
                f.write('duration loss: ' + str(loss_dur) + '\r\n')

            print('\r\n' + 'epoch: ' + str(epoch))
            print('beat: ')
            print(str(beat_results))
            print('duration: ')
            print('train_loss:' + str(loss_dur))
            print('pitch: ')
            print(str(pitch_results))

            if (epoch + 1) % 10 == 0:
                # save backbone
                model = self.backbone
                model.eval()
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = './models/MTL_backbone_' + str(epoch + 1) + '.pt'
                output_config_file = './models/MTL_backbone_' + str(epoch + 1) + '.json'
                torch.save(model_to_save.state_dict(), output_model_file)
                # model_to_save.config.to_json_file(output_config_file)

                # save beat head
                model = self.beat_head
                model.eval()
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = './models/MTL_beat_head_' + str(epoch + 1) + '.pt'
                torch.save(model_to_save.state_dict(), output_model_file)

                # save duration head
                model = self.duration_head
                model.eval()
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = './models/MTL_duration_head_' + str(epoch + 1) + '.pt'
                torch.save(model_to_save.state_dict(), output_model_file)

                # save pitch head
                model = self.pitch_head
                model.eval()
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = './models/MTL_pitch_head_' + str(epoch + 1) + '.pt'
                torch.save(model_to_save.state_dict(), output_model_file)

            if validation_set:
                self.validate(validation_set)

    def validate(self, validation_set):
        val_sampler = torch.utils.data.distributed.DistributedSampler(validation_set)
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=batch_size, sampler=val_sampler, shuffle=False, **kwargs)

        self.backbone.eval()
        self.beat_model.eval()
        self.duration_model.eval()
        self.pitch_model.eval()

        labels_pred = []
        labels_true = []
        durs_pred = []
        durs_true = []
        pitches_pred = []
        pitches_true = []

        for item in tqdm(validation_loader, desc='Validating'):
            # for i, item in enumerate(validation_loader):
            input_ids = item['input_ids'].numpy().tolist()
            input_ids = torch.tensor(input_ids).to(self.device)
            attention_mask = item['attention_mask'].numpy().tolist()
            attention_mask = torch.tensor(attention_mask).to(self.device)

            # duration
            outputs = self.duration_model([input_ids, attention_mask])
            outputs = torch.squeeze(outputs, 1)
            tmp1 = []
            for ite in outputs.cpu().detach().numpy().tolist():
                tmp1.extend(ite)
            durs_pred.append(tmp1)
            tmp = []
            for ite in item['durations'].cpu().detach().numpy().tolist():
                tmp.extend(ite)
            durs_true.append(tmp)

            # beat
            outputs = self.beat_model([input_ids, attention_mask])
            outputs = torch.squeeze(outputs, 1)
            tmp1 = []
            for ite in np.argmax(outputs.cpu().detach().numpy().tolist(), axis=-1):
                tmp1.extend(ite)
            labels_pred.append(tmp1)
            tmp = []
            for ite in item['labels'].cpu().detach().numpy().tolist():
                tmp.extend(ite)
            labels_true.append(tmp)

            # pitch
            outputs = self.pitch_model([input_ids, attention_mask])
            outputs = torch.squeeze(outputs, 1)
            tmp1 = []
            for ite in np.argmax(outputs.cpu().detach().numpy().tolist(), axis=-1):
                tmp1.extend(ite)
            pitches_pred.append(tmp1)
            tmp = []
            for ite in item['pitches'].cpu().detach().numpy().tolist():
                tmp.extend(ite)
            pitches_true.append(tmp)

        true_durations_predictions = [
            [p for (p, d) in zip(prediction, duration) if d != 0]
            for prediction, duration in zip(durs_pred, durs_true)
        ]
        true_durations = [
            [d for (p, d) in zip(prediction, duration) if d != 0]
            for prediction, duration in zip(durs_pred, durs_true)
        ]

        loss_dur = 0
        for i in range(len(true_durations)):
            loss_dur += mean_squared_error(true_durations[i], true_durations_predictions[i])
        loss_dur /= len(true_durations)

        true_labels_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(labels_pred, labels_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(labels_pred, labels_true)
        ]
        beat_results = metric.compute(predictions=true_labels_predictions, references=true_labels, suffix=True)

        true_pitches_predictions = [
            ['L{}-P'.format(p) for (p, l) in zip(prediction, pitch) if l != -100]
            for prediction, pitch in zip(pitches_pred, pitches_true)
        ]
        true_pitches = [
            ['L{}-P'.format(l) for (p, l) in zip(prediction, pitch) if l != -100]
            for prediction, pitch in zip(pitches_pred, pitches_true)
        ]

        # binary classification score (pitch)
        array_pred = np.array([])
        array_true = np.array([])
        for prediction, pitch in zip(pitches_pred, pitches_true):
            tmp_p = []
            tmp_t = []
            for (p, l) in zip(prediction, pitch):
                if l != -100:
                    tmp_p.append(p)
                    tmp_t.append(l)
            array_pred = np.append(array_pred, np.array(tmp_p))
            array_true = np.append(array_true, np.array(tmp_t))
        ave_pred = np.mean(array_pred)
        ave_true = np.mean(array_true)
        std_pred = np.std(array_pred)
        std_true = np.std(array_true)
        pitch_binary_true = []
        pitch_binary_pred = []
        for item in array_pred:
            if item < ave_pred - 1.5 * std_pred:
                pitch_binary_pred.append(-1)
                continue
            if item > ave_pred + 1.5 * std_pred:
                pitch_binary_pred.append(1)
                continue
            pitch_binary_pred.append(0)

        for item in array_true:
            if item < ave_true - 1.5 * std_true:
                pitch_binary_true.append(-1)
                continue
            if item > ave_true + 1.5 * std_true:
                pitch_binary_true.append(1)
                continue
            pitch_binary_true.append(0)

        assert len(pitch_binary_pred) == len(pitch_binary_true)
        pitch_length = len(pitch_binary_pred)
        alto_true = 0
        normal_true = 0
        basso_true = 0
        alto_pred = 0
        normal_pred = 0
        basso_pred = 0
        alto_right = 0
        basso_right = 0
        normal_right = 0
        for p, t in zip(pitch_binary_pred, pitch_binary_true):
            if p == t == 0:
                normal_true += 1
                normal_pred += 1
                normal_right += 1
                continue
            if p == t == 1:
                alto_true += 1
                alto_pred += 1
                alto_right += 1
                continue
            if p == t == -1:
                basso_true += 1
                basso_pred += 1
                basso_right += 1
                continue
            if p == 1:
                alto_pred += 1
            if p == 0:
                normal_pred += 1
            if p == -1:
                basso_pred += 1
            if t == 1:
                alto_true += 1
            if t == 0:
                normal_true += 1
            if t == -1:
                basso_true += 1

        true_tones_predictions = []
        true_tones = []
        for p, l in zip(pitch_binary_pred, pitch_binary_true):
            true_tones_predictions.append('T{}-P'.format(p))
            true_tones.append('T{}-P'.format(l))

        tone_results = metric.compute(predictions=[true_tones_predictions], references=[true_tones], suffix=True)
        pitch_results = metric.compute(predictions=true_pitches_predictions, references=true_pitches, suffix=True)

        with open(record_file, 'a') as f:
            f.write('val: ' + '\r\n')
            f.write('beat score: ' + '\r\n' + str(beat_results) + '\r\n')
            f.write('duration loss: ' + str(loss_dur) + '\r\n' + '\r\n')
            f.write('pitch score: ' + '\r\n' + str(pitch_results) + '\r\n')
            f.write('binary pitch score: ' + '\r\n' + str(tone_results) + '\r\n')
            f.write('pitch statistic: ' + '\r\n')
            f.write('length ={} '.format(pitch_length) + '\r\n')
            f.write('normal_true = {} '.format(normal_true) + 'normal_pred = {} '.format(
                normal_pred) + 'normal_right = {} '.format(normal_right) + '\r\n')
            f.write('alto_true = {} '.format(alto_true) + 'alto_pred = {} '.format(
                alto_pred) + 'alto_right = {} '.format(alto_right) + '\r\n')
            f.write('basso_true = {} '.format(basso_true) + 'basso_pred = {} '.format(
                basso_pred) + 'basso_right = {} '.format(basso_right) + '\r\n')

        print('beat: ')
        print(str(beat_results))
        print('duration: ')
        print('val_loss:' + str(loss_dur))
        print('pitch: ')
        print(str(pitch_results))
        print('binary pitch: ')
        print(str(tone_results))
