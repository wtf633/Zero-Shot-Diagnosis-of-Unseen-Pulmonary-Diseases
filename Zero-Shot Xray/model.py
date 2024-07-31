rom pathlib import Path
from typing import List
import re

import torch
import torch.nn.functional as F
from health_multimodal.image import get_biovil_resnet_inference
from health_multimodal.text import get_cxr_bert_inference
from health_multimodal.vlp import ImageTextInferenceEngine

from utils import cos_sim_to_prob, prob_to_log_prob, log_prob_to_prob


class InferenceModel():
    def __init__(self):
        self.text_inference = get_cxr_bert_inference()
        self.image_inference = get_biovil_resnet_inference()
        self.image_text_inference = ImageTextInferenceEngine(
            image_inference_engine=self.image_inference,
            text_inference_engine=self.text_inference,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_text_inference.to(self.device)

        # caches for faster inference
        self.text_embedding_cache = {}
        self.image_embedding_cache = {}

        self.transform = self.image_inference.transform

    def get_similarity_score_from_raw_data(self, image_embedding, image_embedding1, query_text: str) -> float:
        """Compute the cosine similarity score between an image and one or more strings.
        If multiple strings are passed, their embeddings are averaged before L2-normalization.
        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :return: The similarity score between the image and the text.
        """
        assert not self.image_text_inference.image_inference_engine.model.training
        assert not self.image_text_inference.text_inference_engine.model.training
        if query_text in self.text_embedding_cache:
            text_embedding = self.text_embedding_cache[query_text]
        else:
            text_embedding = self.image_text_inference.text_inference_engine.get_embeddings_from_prompt([query_text],
                                                                                                        normalize=False)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding = F.normalize(text_embedding, dim=0, p=2)
            self.text_embedding_cache[query_text] = text_embedding

        cos_similarity = image_embedding @ text_embedding.t()
        return cos_similarity.item()

    def process_image(self, image):
        ''' same code as in image_text_inference.image_inference_engine.get_projected_global_embedding() but adapted
        to deal with image instances instead of path'''

        transformed_image = self.transform(image)
        projected_img_emb = self.image_inference.model.forward(transformed_image).projected_global_embedding
        projected_img_emb = F.normalize(projected_img_emb, dim=-1)
        assert projected_img_emb.shape[0] == 1
        assert projected_img_emb.ndim == 2
        return projected_img_emb[0]

    def get_descriptor_probs(self, image_path: Path, descriptors: List[str]):
        probs = {}
        negative_probs = {}
        image_embedding = self.image_text_inference.image_inference_engine.get_projected_global_embedding(image_path)
        image_embedding1, img_shape = self.image_text_inference.image_inference_engine.get_projected_patch_embeddings(
            image_path)

        descriptions_NLM = [
            "",
            "within normal limits. ",
            "in size. ",
            "in place. ",
            "on the left. ",
            "on the right. ",
            "in the right lung. ",
            "in the left lung. ",
            "in the right upper lobe. ",
            "in the left upper lobe. ",
            "in the right lower lobe. ",
            "in the left lower lobe. ",
            "in the right mid lung. ",
            "in the left mid lung. ",
            "in the right lung base. ",
            "in the left lung base. ",
            "at the lung bases. ",
            "at both lung bases. ",
            "in the right upper lung. ",
            "in the left upper lung. ",
            "below the diaphragm. ",
            "in the right lower lung. ",
            "in the left lower lung. ",
        ]

        descriptions_nodes = [
            "",
            "in size. ",
            "in place. ",
            "in appearance. ",
            "on the left. ",
            "on the right. ",
            "in the right lung. ",
            "in the left lung. ",
            "in the right upper lobe. ",
            "in the left upper lobe. ",
            "in the right lower lobe. ",
            "in the left lower lobe. ",
            "in the right mid lung. ",
            "in the left mid lung. ",
            "in the right upper lung. ",
            "in the left upper lung. ",
        ]

        descriptions_COVID = [
            "",
            "within normal limits. ",
            "in place. ",
            "in position. ",
            "in the right lung. ",
            "in the left lung. ",
            "in the right upper lobe. ",
            "in the left upper lobe. ",
            "in the right lower lobe. ",
            "in the left lower lobe. ",
            "in the right mid lung. ",
            "in the left mid lung. ",
            "in the right lung base. ",
            "in the left lung base. ",
            "at the lung bases. ",
            "at both lung bases. ",
            "in the right upper lung. ",
            "in the left upper lung. ",
        ]

        txt = []
        str = ""
        for desc in descriptors:
            # This is an X-ray image.
            # An X-ray image of a lung.
            # An X-ray image.
            # An X-ray image of the lungs.
            # A lung X-ray image.
            # This is an X-ray image of a lung.
            # This is a lung X-ray image.
            # This is an X-ray image of the lungs.
            str1 = "No Finding"
            if str1 in desc:
                # prompt = f'There are {desc}'
                # prompt = f'{desc}'
                prompt = f'An X-ray image of the lungs. Findings, there are {desc}'
                score = self.get_similarity_score_from_raw_data(image_embedding, image_embedding1, prompt)
                # print(prompt)
            else:
                score = 0
                for de in descriptions_nodes:  # descriptions_NLM descriptions_nodes descriptions_COVID
                    prompt = f'An X-ray image of the lungs. Findings, there are {desc}'
                    t = f' {de}Impression,'
                    if de == "":
                        full_description = prompt
                    else:
                        full_description = prompt.replace(". Impression,", t)
                    # print(full_description)
                    s = self.get_similarity_score_from_raw_data(image_embedding, image_embedding1, full_description)
                    if abs(s) > abs(score):
                        score = s
                        str = full_description

            neg_prompt = f'An X-ray image of the lungs. Findings, there are no {desc}'
            # neg_prompt = f'No {desc}'
            # print(neg_prompt)
            neg_score = self.get_similarity_score_from_raw_data(image_embedding, image_embedding1, neg_prompt)
            pos_prob, neg_prob = torch.softmax((torch.tensor([score, neg_score]) / 0.5), dim=0)
            negative_probs[desc] = neg_prob

            probs[desc] = pos_prob
            if pos_prob > 0.5:
                if str1 in desc:
                    txt.append(desc.split('.')[0])
                else:
                    start = str.find('there are') + len('there are')
                    end = str.find('. Impression')
                    result = str[start:end].strip()
                    txt.append(result)
        return probs, negative_probs, txt

        # for desc in descriptors:
        #     # prompt = f'An X-ray image of a lung. There are {desc}'
        #     # prompt = f'a photo of lung. There are {desc}'
        #     # print(desc)
        #     # prompt = f'There are {desc}'
        #     prompt = f'{desc}'
        #     # prompt = f'An X-ray image. There are {desc}'
        #     # print(prompt)
        #     score = self.get_similarity_score_from_raw_data(image_embedding, image_embedding1, prompt)
        #     if do_negative_prompting:
        #         # neg_prompt = f'An X-ray image of a lung. There are no {desc}'
        #         # neg_prompt = f'a photo of lung. There are no {desc}'
        #         # neg_prompt = f'There are no {desc}'
        #         neg_prompt = f'No {desc}'
        #         # neg_prompt = f'An X-ray image. There are no {desc}'
        #         neg_score = self.get_similarity_score_from_raw_data(image_embedding, image_embedding1, neg_prompt)
        #     pos_prob = cos_sim_to_prob(score)
        #
        #     if do_negative_prompting:
        #         pos_prob, neg_prob = torch.softmax((torch.tensor([score, neg_score]) / 0.5), dim=0)
        #         negative_probs[desc] = neg_prob
        #
        #     probs[desc] = pos_prob
        #
        # return probs, negative_probs

    # Overall, the X-ray results indicate that the patient's lungs and heart are functioning normally
    # The findings indicate
    def get_all_descriptors(self, disease_descriptors):
        all_descriptors = set()
        for disease, descs in disease_descriptors.items():
            all_descriptors.update([f"{desc}. Impression, {disease}." for desc in descs])
            # all_descriptors.update([f"{desc}" for desc in descs])
        all_descriptors = sorted(all_descriptors)
        return all_descriptors

    # def get_all_descriptors_only_disease(self, disease_descriptors):
    #     print("A" * 100)
    #     all_descriptors = set()
    #     for disease, descs in disease_descriptors.items():
    #         all_descriptors.update([f"{desc}" for desc in descs])
    #     all_descriptors = sorted(all_descriptors)
    #     return all_descriptors

    def get_diseases_probs(self, disease_descriptors, pos_probs, negative_probs, prior_probs=None,
                           do_negative_prompting=True):
        # print(pos_probs)
        # print(negative_probs)
        disease_probs = {}
        disease_neg_probs = {}
        for disease, descriptors in disease_descriptors.items():
            desc_log_probs = []
            desc_neg_log_probs = []
            for desc in descriptors:
                desc = f"{desc}. Impression, {disease}."
                # desc = f"{desc}"
                desc_log_probs.append(prob_to_log_prob(pos_probs[desc]))
                if do_negative_prompting:
                    desc_neg_log_probs.append(prob_to_log_prob(negative_probs[desc]))
            # print("pos_probs:",pos_probs)
            # print("negative_probs:",negative_probs)
            disease_log_prob = sum(sorted(desc_log_probs, reverse=True)) / len(desc_log_probs)
            if do_negative_prompting:
                disease_neg_log_prob = sum(desc_neg_log_probs) / len(desc_neg_log_probs)
            disease_probs[disease] = log_prob_to_prob(disease_log_prob)
            if do_negative_prompting:
                disease_neg_probs[disease] = log_prob_to_prob(disease_neg_log_prob)
            # print("disease_probs",disease_probs)
            # print("disease_neg_probs",disease_neg_probs)

        return disease_probs, disease_neg_probs

    # Threshold Based
    def get_predictions(self, disease_descriptors, threshold, disease_probs, keys):
        predicted_diseases = []
        prob_vector = torch.zeros(len(keys), dtype=torch.float)  # num of diseases
        for idx, disease in enumerate(disease_descriptors):
            if disease == 'No Finding':
                continue
            prob_vector[keys.index(disease)] = disease_probs[disease]
            if disease_probs[disease] > threshold:
                predicted_diseases.append(disease)

        if len(predicted_diseases) == 0:  # No finding rule based
            prob_vector[0] = 1.0 - max(prob_vector)
        else:
            prob_vector[0] = 1.0 - max(prob_vector)

        return predicted_diseases, prob_vector

    # Negative vs Positive Prompting
    def get_predictions_bin_prompting(self, disease_descriptors, disease_probs, negative_disease_probs, keys, txt):
        predicted_diseases = []
        prob_vector = torch.zeros(len(keys), dtype=torch.float)  # num of diseases
        for idx, disease in enumerate(disease_descriptors):
            if disease == 'No Finding':
                continue
            pos_neg_scores = torch.tensor([disease_probs[disease], negative_disease_probs[disease]])
            # print("pos_neg_scores:",pos_neg_scores)
            prob_vector[keys.index(disease)] = pos_neg_scores[0]
            if torch.argmax(pos_neg_scores) == 0:  # Positive is More likely
                predicted_diseases.append(disease)

        print(predicted_diseases)
        if len(predicted_diseases) == 0:  # No finding rule based
            prob_vector[0] = 1.0 - max(prob_vector)
        else:
            prob_vector[0] = 1.0 - max(prob_vector)

        if prob_vector[0]>0.5 or prob_vector[1]>0.5:
            print("分数：", prob_vector)
            txt = ', '.join(txt)
            if len(predicted_diseases) == 0:
                string = f"An X-ray image of the lungs. Findings, there are {txt}. Impression, No Finding."
            else:
                string = f"An X-ray image of the lungs. Findings, there are {txt}. Impression, {predicted_diseases[0]}."

            # 将每句话开头的第一个字母大写，其他都小写
            def capitalize_sentence(sentence):
                sentence = sentence.strip()
                if sentence:
                    return sentence[0].upper() + sentence[1:].lower()
                return sentence

            sentences = re.split(r'(?<=\.)\s*', string)
            result = '. '.join(capitalize_sentence(sentence) for sentence in sentences if sentence)
            result = result.replace('..', '.')
            print(result)
        return predicted_diseases, prob_vector
