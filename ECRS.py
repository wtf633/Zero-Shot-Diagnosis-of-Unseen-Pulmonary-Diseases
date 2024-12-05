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
