import re
import pandas as pd
import torch
from torch.utils.data import Dataset


class NLMDataset1(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/NLM1/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/NLM1/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'Tuberculosis']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class NLMDataset2(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/NLM2/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/NLM2/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'Tuberculosis']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class JSRTDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/JSRT/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/JSRT/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'Nodule']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class COVIDDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/COVID/test_list1.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/COVID/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            # disease_keys = ['No Finding', 'COVID-19', 'Pneumonia']
            disease_keys = ['No Finding', 'COVID-19']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class COVIDXDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/COVIDX/test_list1.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/COVIDX/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            # disease_keys = ['No Finding', 'COVID-19', 'Pneumonia']
            disease_keys = ['No Finding', 'COVID-19']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class COVID19Dataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/COVID19/test_list2.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/COVID19/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            # disease_keys = ['No Finding', 'COVID-19', 'Lung Opacity', 'Pneumonia']
            disease_keys = ['No Finding', 'COVID-19']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys


class HisNDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        test_image_names = set()
        with open('data/HisN/test_list.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                test_image_names.add(line.strip())
        self.meta_infos = []
        for idx, d in data.iterrows():
            if d.Path not in test_image_names:
                continue
            image_path = f'data/HisN/images_merge/{d.Path}'
            diseases = {}
            # Hardcoded for robustness and consistency
            disease_keys = ['No Finding', 'Nodule']
            for key, value in d.iteritems():
                key = key.replace('_', ' ')
                if key in disease_keys:
                    diseases[key] = value
            disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
            self.meta_infos.append(
                {'image_path': image_path, 'disease_keys': disease_keys, 'disease_values': disease_values})

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, idx):
        meta_info = self.meta_infos[idx]
        image_path = meta_info['image_path']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_path, labels, keys