# Zero-Shot-Diagnosis-of-Unseen-Pulmonary-Diseases
In this work, we first balanced the brightness and contrast of the images based on their feature distribution and designed two sets of multi-scale filters to restore lesion details and enhance
contour features. For text data, we utilized ChatGPT-4o to extract feature descriptions of three unseen diseases, which were then refined by experienced radiologists. We extracted information
on lesion location, shape, and report template styles from numerous lung clinical diagnostic reports to provide more detailed descriptions. Finally, we computed the contrast probability between the visual and textual modalities using cosine similarity. Furthermore, our method was evaluated on two tuberculosis datasets, two lung nodule datasets (one private), and three COVID-19 datasets. Compared to advanced methods, our approach showed significant advantages on small sample datasets, even surpassing some supervised learning methods. This underscores its clinical significance in diagnosing unseen and rare diseases.

#### Conceptual Diagram of Zero-Shot Learning Application in Pulmonary X-Ray Diagnosis (e.g., COVID-19).
<img src="https://github.com/wtf633/Zero-Shot-Diagnosis-of-Unseen-Pulmonary-Diseases/blob/main/Conceptual%20Diagram.jpg" alt="示例图片" width="600">

#### Conceptual Diagram of Zero-Shot Learning Application in Pulmonary X-Ray Diagnosis (e.g., COVID-19).
<img src="https://github.com/wtf633/Zero-Shot-Diagnosis-of-Unseen-Pulmonary-Diseases/blob/main/Flowchart%20of%20the%20Proposed%20Zero-Shot%20Method.jpg" alt="示例图片" width="600">
