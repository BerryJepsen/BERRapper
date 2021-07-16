import torch
import datetime as dt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}

# batch_size = 48
batch_size = 32

# parameters
dataset_path = ['./Src/final_data.pickle']
bert_model_path = './pretrained_models/bert-base-chinese.bin'
bert_config_path = './pretrained_models/config.json'

task = "multi-tasks"
model_name = "bert-base-chinese"
now_time = dt.datetime.now().strftime('%F %T')
record_file = './Out/' + str(now_time) + '_' + task + '.txt'
val_portion = 0.2
training_epoch = 30

label_all_tokens = True

label_list = [
    'O-N',  # Ordinary label
    'S1-P',  # Stressed word(Beat)
    'S2-P',  # Stressed word(Beat)
    'S3-P',  # Stressed word(Beat)
    'S4-P',  # Stressed word(Beat)
    'S5-P',  # Stressed word(Beat)
    'S6-P',  # Stressed word(Beat)
    'S7-P'  # Stressed word(Beat)
]
