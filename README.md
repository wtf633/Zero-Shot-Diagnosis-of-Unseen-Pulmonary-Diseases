# Zero-Shot-Diagnosis-of-Unseen-Pulmonary-Diseases
In this work, we first balanced the brightness and contrast of the images based on their feature distribution and designed two sets of multi-scale filters to restore lesion details and enhance
contour features. For text data, we utilized ChatGPT-4o to extract feature descriptions of three unseen diseases, which were then refined by experienced radiologists. We extracted information
on lesion location, shape, and report template styles from numerous lung clinical diagnostic reports to provide more detailed descriptions. Finally, we computed the contrast probability between the visual and textual modalities using cosine similarity. Furthermore, our method was evaluated on two tuberculosis datasets, two lung nodule datasets (one private), and three COVID-19 datasets. Compared to advanced methods, our approach showed significant advantages on small sample datasets, even surpassing some supervised learning methods. This underscores its clinical significance in diagnosing unseen and rare diseases.

#### Conceptual Diagram of Zero-Shot Learning Application in Pulmonary X-Ray Diagnosis (e.g., COVID-19).
<img src="https://github.com/wtf633/Zero-Shot-Diagnosis-of-Unseen-Pulmonary-Diseases/blob/main/Conceptual%20Diagram.jpg" alt="示例图片" width="600">

### Description of experimental datasets
#### Tuberculosis
1) MCXD includes 58 tuberculosis and 80 normal chest X-rays in resolutions of 4020×4892 or 4892×4020 pixels, created by the U.S. National Library of Medicine. (https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)
2) SCXD has 336 tuberculosis and 326 normal chest X-rays at 3K×3K pixels, developed by the Shenzhen’s Third People’s Hospital, Guangdong Medical College, China. (https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)
#### Nodules
3) JSRT contains 154 nodule and 93 non-nodule chest X-rays at 2048×2048 pixels, created by the Japanese Radiological Society and American medical institutions. (http://db.jsrt.or.jp/eng.php) or (https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt)
4) HCXD is a private dataset from Hefei Cancer Hospital containing 113 lung nodule and 200 normal images collected between January 1, 2023, and February 1, 2024.
#### COVID-19
5) COVID-QED has 33,920 chest Xray images: 11,956 COVID-19, 11,263 non-COVID infections, and 10,701 normal cases. We used COVID-19 segmented data at 256×256 pixels for evaluation.
6) COVID-19-CXD comprises 21,165 chest X-rays, including 3,616 COVID-19 positive, 6,012 lung opacity, 1,345 viral pneumonia, and 10,192 normal cases, at 299×299 pixels, developed by Qatar
University, Dhaka University, and physicians from Pakistan.
7) COVIDx includes 84,818 chest X-rays from 45,342 subjects, curated by the Vision and Image Processing Research Group at the University of Waterloo, Canada, evaluated using COVIDx9B test data at 1024×1024 pixels.
