from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn.functional as F
import torch.nn as nn
from MTL_hparam import model_name, device, bert_model_path, bert_config_path

backbone = BertModel.from_pretrained(model_name)

# config = BertConfig.from_json_file(bert_config_path)
# backbone = BertModel(config)

#config = BertConfig.from_json_file(bert_config_path)
#backbone = BertModel.from_pretrained(bert_model_path, config=config)

class Backbone(torch.nn.Module):
    def init(self):
        super(Backbone, self).__init__
        self.backbone = backbone

    def forward(self, x):
        hidden = self.backbone(x[0], attention_mask=x[1])[0]
        return hidden


class Beat_Head(torch.nn.Module):
    def init(self):
        super(Beat_Head, self).__init__
        self.fc0 = nn.Linear(768, 60)
        self.fc1 = nn.Linear(60, 8)

    def forward(self, hidden):  # attention mask
        net = self.fc0(hidden)
        net = F.leaky_relu(net)
        net = self.fc1(net)
        sm = nn.Softmax(dim=1)
        net = sm(net)
        return net


class Duration_Head(torch.nn.Module):
    def init(self):
        super(Duration_Head, self).__init__
        self.conv0 = nn.Sequential(
            nn.Conv1d(768, 64, 3, stride=1, padding=1),  # 768 to 64
            nn.LeakyReLU(),
            nn.BatchNorm1d(64).to(device),
            # nn.Dropout(p=0.5).to(device)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 1, 2, stride=1, padding=1),  # 64 to 1
            nn.LeakyReLU(),
            # nn.BatchNorm1d(1).to(device),
            # nn.Dropout(p=0.5).to(device)
        )
        self.fc = nn.Linear(129, 128)

    def forward(self, hidden):  # attention mask
        # hidden: (batch_size, 128, 768)
        hidden = hidden.view(-1, 768, 128)
        # hidden: (batch_size, 768, 128)
        net = self.conv0(hidden)
        # net: (batch_size, 64, 128)
        net = self.conv1(net)
        # net: (batch_size, 1, 129)
        net = self.fc(net)
        # net: (batch_size, 1, 128)
        return net


class Pitch_Head(torch.nn.Module):
    def init(self):
        super(Pitch_Head, self).__init__
        self.conv0 = nn.Sequential(
            nn.Conv1d(768, 128, 3, stride=1, padding=1),  # 768 to 128
            nn.LeakyReLU(),
            nn.BatchNorm1d(128).to(device),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 48, 2, stride=1, padding=1),  # 128 to 48
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(129, 128)

    def forward(self, hidden):  # attention mask
        # hidden: (batch_size, 128, 768)
        hidden = hidden.view(-1, 768, 128)
        # hidden: (batch_size, 768, 128)
        net = self.conv0(hidden)
        # net: (batch_size, 128, 128)
        net = self.conv1(net)
        # net: (batch_size, 48, 129)
        net = self.fc(net)
        # net: (batch_size, 48, 128)
        net = net.view(-1, 128, 48)
        # net: (batch_size, 128, 48)
        sm = nn.Softmax(dim=1)
        net = sm(net)
        return net
