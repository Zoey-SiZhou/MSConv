from easydict import EasyDict as edict
import datetime

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "ms50"
config.resume = False
config.save_all_states = False
config.output = r"xxxxxxxxxxx" # Path of result
config.embedding_size = 512

# Partial FC
config.sample_rate = 1.0  # 0.9
config.interclass_filtering_threshold = 0
config.fp16 = False  # True
config.batch_size = 128

# For SGD
config.optimizer = "sgd"
config.lr = 0.02  # 0.006
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 1e-4  # 0.1

config.verbose = 16000
config.frequent = 10

# For Large Sacle Dataset, such as WebFace42M
config.dali = False
config.dali_aug = False

# Gradient ACC
config.gradient_acc = 1

# data-load num_workers
config.num_workers = 4
# setup seed
config.seed = 2048

config.rec = r"xxxxx"  # Path of training set
config.val = r"xxxxxxxxxx"  # Path of validation set
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', 'calfw', 'cplfw', 'agedb_30', 'vgg2_fp']

# WandB Logger
config.using_wandb = False
# config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
# config.suffix_run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
config.wandb_entity = "XXXX"
config.wandb_project = "XXXX"
config.wandb_log_all = True
config.save_artifacts = True
config.wandb_resume = False
# resume wandb run: Only if you wand t resume the last run that it was interrupted