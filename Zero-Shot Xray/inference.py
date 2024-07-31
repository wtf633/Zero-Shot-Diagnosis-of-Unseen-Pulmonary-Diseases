import argparse
import gc
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from chestxray import COVIDDataset, JSRTDataset, NLMDataset1, NLMDataset2, COVIDXDataset, COVID19Dataset, HisNDataset
from descriptors import disease_descriptors_NLM, disease_descriptors_COVID, disease_descriptors_JSRT
from model import InferenceModel
from utils import calculate_auroc, calculate_total
import warnings

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def inference_NLM1():
    dataset = NLMDataset1(f'data/NLM1/NLM.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_NLM)
    print(len(dataloader))
    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        probs, negative_probs, txt = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_NLM,
                                                                                   pos_probs=probs,
                                                                                   negative_probs=negative_probs)

        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
            disease_descriptors_NLM,
            disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs,
            keys=keys,
            txt=txt
        )
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    df_labels_clean = pd.DataFrame(all_labels_clean)
    df_probs_neg_clean = pd.DataFrame(all_probs_neg_clean)
    df_labels_clean.to_csv('labels_NLM1.csv', index=False)
    df_probs_neg_clean.to_csv('probs_NLM1.csv', index=False)
    calculate_total(df_labels_clean, df_probs_neg_clean)

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')
    print("-" * 30)
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_NLM2():
    dataset = NLMDataset2(f'data/NLM2/NLM.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_NLM)
    print(len(dataloader))
    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        probs, negative_probs, txt = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_NLM,
                                                                                   pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
            disease_descriptors_NLM,
            disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs,
            keys=keys,
            txt=txt
        )
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    df_labels_clean = pd.DataFrame(all_labels_clean)
    df_probs_neg_clean = pd.DataFrame(all_probs_neg_clean)
    df_labels_clean.to_csv('labels_NLM2.csv', index=False)
    df_probs_neg_clean.to_csv('probs_NLM2.csv', index=False)
    calculate_total(df_labels_clean, df_probs_neg_clean)

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')
    print("-" * 30)
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_JSRT():
    dataset = JSRTDataset(f'data/JSRT/jsrt_metadata.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_JSRT)
    # print(all_descriptors)
    print(len(dataloader))
    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        # print(image_path)
        probs, negative_probs, txt = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        # print("probs",probs)
        # print("negative_probs",negative_probs)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_JSRT,
                                                                                   pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
            disease_descriptors_JSRT,
            disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs,
            keys=keys,
            txt=txt
        )

        # print("-"*100)
        # print(predicted_diseases)
        # print(prob_vector_neg_prompt)
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    df_labels_clean = pd.DataFrame(all_labels_clean)
    df_probs_neg_clean = pd.DataFrame(all_probs_neg_clean)
    df_labels_clean.to_csv('labels_JSRT.csv', index=False)
    df_probs_neg_clean.to_csv('probs_JSRT.csv', index=False)
    calculate_total(df_labels_clean, df_probs_neg_clean)

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')
    print("-" * 30)
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_COVID():
    dataset = COVIDDataset(f'data/COVID/COVID_filter.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_COVID)
    print(len(dataloader))
    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        probs, negative_probs, txt = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_COVID,
                                                                                   pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
            disease_descriptors_COVID,
            disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs,
            keys=keys,
            txt=txt
        )

        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    df_labels_clean = pd.DataFrame(all_labels_clean)
    df_probs_neg_clean = pd.DataFrame(all_probs_neg_clean)
    # df_labels_clean.to_csv('labels_COVID.csv', index=False)
    # df_probs_neg_clean.to_csv('probs_COVID.csv', index=False)
    calculate_total(df_labels_clean, df_probs_neg_clean)

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')
    print("-" * 30)
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_COVIDX():
    dataset = COVIDXDataset(f'data/COVIDX/COVID_Data_filter.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_COVID)
    print(len(dataloader))
    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        # print(image_path)
        probs, negative_probs, txt = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        # print("probs",probs)
        # print("negative_probs",negative_probs)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_COVID,
                                                                                   pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
            disease_descriptors_COVID,
            disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs,
            keys=keys,
            txt=txt
        )

        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    df_labels_clean = pd.DataFrame(all_labels_clean)
    df_probs_neg_clean = pd.DataFrame(all_probs_neg_clean)
    # df_labels_clean.to_csv('labels_COVIDX.csv', index=False)
    # df_probs_neg_clean.to_csv('probs_COVIDX.csv', index=False)
    calculate_total(df_labels_clean, df_probs_neg_clean)

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')
    print("-" * 30)
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_COVID19():
    dataset = COVID19Dataset(f'data/COVID19/metadata_filter.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_COVID)
    print(len(dataloader))
    all_labels = []
    all_probs_neg = []
    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        # print(image_path)
        probs, negative_probs, txt = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_COVID,
                                                                                   pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
            disease_descriptors_COVID,
            disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs,
            keys=keys,
            txt=txt
        )
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)

    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]

    df_labels_clean = pd.DataFrame(all_labels_clean)
    df_probs_neg_clean = pd.DataFrame(all_probs_neg_clean)
    # df_labels_clean.to_csv('labels_COVID19.csv', index=False)
    # df_probs_neg_clean.to_csv('probs_COVID19.csv', index=False)
    calculate_total(df_labels_clean, df_probs_neg_clean)

    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')
    print("-" * 30)
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


def inference_HisN():
    dataset = HisNDataset(f'/home/image023/data/Xplainer-master/data/HisN/HisN.csv')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=1)
    inference_model = InferenceModel()
    all_descriptors = inference_model.get_all_descriptors(disease_descriptors_JSRT)
    # print(all_descriptors)
    print(len(dataloader))
    all_labels = []
    all_probs_neg = []

    for batch in tqdm(dataloader):
        batch = batch[0]
        image_path, labels, keys = batch
        image_path = Path(image_path)
        # print(image_path)
        probs, negative_probs, txt = inference_model.get_descriptor_probs(image_path, descriptors=all_descriptors)
        # print("probs",probs)
        # print("negative_probs",negative_probs)
        disease_probs, negative_disease_probs = inference_model.get_diseases_probs(disease_descriptors_JSRT,
                                                                                   pos_probs=probs,
                                                                                   negative_probs=negative_probs)
        # print("*" * 100)
        # print(disease_probs)
        # print("*" * 100)
        # print(negative_disease_probs)
        predicted_diseases, prob_vector_neg_prompt = inference_model.get_predictions_bin_prompting(
            disease_descriptors_JSRT,
            disease_probs=disease_probs,
            negative_disease_probs=negative_disease_probs,
            keys=keys,
            txt=txt
        )
        # print("*"*100)
        # print(prob_vector_neg_prompt)
        all_labels.append(labels)
        all_probs_neg.append(prob_vector_neg_prompt)
        gc.collect()

    all_labels = torch.stack(all_labels)
    all_probs_neg = torch.stack(all_probs_neg)
    #
    existing_mask = sum(all_labels, 0) > 0
    all_labels_clean = all_labels[:, existing_mask]
    all_probs_neg_clean = all_probs_neg[:, existing_mask]
    all_keys_clean = [key for idx, key in enumerate(keys) if existing_mask[idx]]
    #
    df_labels_clean = pd.DataFrame(all_labels_clean)
    df_probs_neg_clean = pd.DataFrame(all_probs_neg_clean)
    # df_labels_clean.to_csv('labels_HisN.csv', index=False)
    # df_probs_neg_clean.to_csv('probs_HisN.csv', index=False)
    calculate_total(df_labels_clean, df_probs_neg_clean)
    #
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean[:, 1:], all_labels_clean[:, 1:])
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean[1:]):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')
    print("-" * 30)
    overall_auroc, per_disease_auroc = calculate_auroc(all_probs_neg_clean, all_labels_clean)
    print(f"AUROC: {overall_auroc:.5f}\n")
    for idx, key in enumerate(all_keys_clean):
        print(f'{key}: {per_disease_auroc[idx]:.5f}')


if __name__ == '__main__':
    print("NLM1")
    inference_NLM1()
    print("ShenZhen")
    inference_NLM2()
    print("JSRT")
    inference_JSRT()
    print("HisN")
    inference_HisN()
    print("COVIDX")
    inference_COVIDX()
    print("COVID19")
    inference_COVID19()
    print("COVID")
    inference_COVID()
