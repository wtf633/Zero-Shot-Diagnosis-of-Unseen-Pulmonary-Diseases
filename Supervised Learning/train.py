import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import pandas as pd
import random
from torchvision import datasets
from torch.backends import cudnn
from torchvision.datasets import FakeData
import warnings


def setup_seed(seed: int):
    """
    全局固定随机种子
    :param seed: 随机种子值
    :return: None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '0,1'
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False


rs = 114514
setup_seed(rs)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                              transforms.RandomRotation(10),
                              transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
                              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                              transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])

test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                             transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])

# 数据集文件夹路径
dataset_dir = "/home/image023/data/code/data"
train_path = os.path.join(dataset_dir, 'COVID19')
test_path = os.path.join(dataset_dir, 'COVIDX')
print('训练集路径', train_path)
print('测试集路径', test_path)

# 载入数据集
train_dataset = datasets.ImageFolder(train_path, train_transform)
test_dataset = datasets.ImageFolder(test_path, test_transform)

print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)

BATCH_SIZE = 8

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True
                          )

# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4
                         )

images, labels = next(iter(train_loader))
images = images.numpy()

# pretrained_cfg = timm.models.create_model('convnext_base').default_cfg
# pretrained_cfg['file'] = r'weight/Convnext_base.bin'
# model = timm.models.convnext_base(pretrained=True, pretrained_cfg=pretrained_cfg)
# model.head.fc = nn.Linear(model.head.fc.in_features, 2)

# pretrained_cfg = timm.models.create_model('vit_base_patch16_224').default_cfg
# pretrained_cfg['file'] = r'weight/Vit_base_patch16_224.bin'
# model = timm.models.vit_base_patch16_224(pretrained=True, pretrained_cfg=pretrained_cfg)
# model.head = nn.Linear(model.head.in_features, 2)

pretrained_cfg = timm.models.create_model('efficientnet_b0').default_cfg
pretrained_cfg['file'] = r'weight/Efficientnet_b0.bin'
model = timm.models.efficientnet_b0(pretrained=True, pretrained_cfg=pretrained_cfg)
model.classifier = nn.Linear(model.classifier.in_features, 2)

print(pretrained_cfg['file'])


class ExponentialMovingAverage(object):
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.backup_params = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow_params
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow_params[name]
                self.shadow_params[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow_params
                self.backup_params[name] = param.data.clone()
                param.data = self.shadow_params[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup_params
                param.data = self.backup_params[name]
        self.backup_params = {}


optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)

# 训练配置
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
ema = ExponentialMovingAverage(model)
ema.register()

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练轮次 Epoch
EPOCHS = 200
# 学习率降低策略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.35)


# 在训练集上训练
def train_one_batch():
    loss_list_train = []
    labels_list_train = []
    preds_list_train = []
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 更新EMA
        ema.update()

        score, preds = torch.max(outputs, 1)

        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        loss_list_train.append(loss)
        labels_list_train.extend(labels)
        preds_list_train.extend(preds)

    # 应用EMA的参数到模型
    ema.apply_shadow()
    log_train = {}
    log_train['epoch'] = epoch

    # 计算分类评估指标
    log_train['train_loss'] = np.mean(loss_list_train)
    log_train['train_accuracy'] = accuracy_score(labels_list_train, preds_list_train)
    log_train['train_precision'] = precision_score(labels_list_train, preds_list_train, average='macro')
    log_train['train_recall'] = recall_score(labels_list_train, preds_list_train)
    log_train['train_f1-score'] = f1_score(labels_list_train, preds_list_train, average='macro')
    log_train['train_roc_auc'] = roc_auc_score(labels_list_train, preds_list_train)

    return log_train


# 在整个测试集上评估
def evaluate_testset():
    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # 获取整个测试集的标签类别和预测类别
            score, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {}
    log_test['epoch'] = epoch

    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list)
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    log_test['test_roc_auc'] = roc_auc_score(labels_list, preds_list)

    return log_test


# 训练开始之前，记录日志
df_train_log = pd.DataFrame()
df_test_log = pd.DataFrame()

# 运行训练
t = 1

for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {epoch}/{EPOCHS}')
    ## 训练阶段
    model.train()
    log_train = {}

    log_train_t = train_one_batch()
    log_train.update(log_train_t)
    df_train_log = pd.concat([df_train_log, pd.DataFrame([log_train])], ignore_index=True)

    lr_scheduler.step()

    ## 测试阶段
    model.eval()
    log_test = {}

    log_test_t = evaluate_testset()
    log_test.update(log_test_t)
    df_test_log = pd.concat([df_test_log, pd.DataFrame([log_test])], ignore_index=True)

    # 恢复模型参数
    ema.restore()

    merged_df = pd.merge(df_train_log, df_test_log, on='epoch')
    merged_df.to_csv('result_COVID19_COVIDX_Efficient.csv', index=False)
