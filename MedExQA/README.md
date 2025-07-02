---
configs:
- config_name: biomedical_engineer
  data_files:
  - split: dev
    path: dev/biomedical_engineer_dev.tsv
  - split: test
    path: test/biomedical_engineer_test.tsv
- config_name: clinical_laboratory_scientist
  data_files:
  - split: dev
    path: dev/clinical_laboratory_scientist_dev.tsv
  - split: test
    path: test/clinical_laboratory_scientist_test.tsv
- config_name: clinical_psychologist
  data_files:
  - split: dev
    path: dev/clinical_psychologist_dev.tsv
  - split: test
    path: test/clinical_psychologist_test.tsv
- config_name: occupational_therapist
  data_files:
  - split: dev
    path: dev/occupational_therapist_dev.tsv
  - split: test
    path: test/occupational_therapist_test.tsv
- config_name: speech_pathologist
  data_files:
  - split: dev
    path: dev/speech_pathologist_dev.tsv
  - split: test
    path: test/speech_pathologist_test.tsv
task_categories:
- multiple-choice
- text-generation
license: cc-by-nc-sa-4.0
language:
- en
tags:
- medical
- biology
size_categories:
- n<1K
---

<h1 align="center">  🧑‍⚕️ MedExQA  </h1>
<h3 align="center"> Medical Question Answering Benchmark with Multiple Explanations </h3>

<p align="center">
 📄 <a href="https://arxiv.org/abs/2406.06331" target="_blank">Paper</a> • ⭐ <a href="https://github.com/knowlab/MedExQA" target="_blank">Code</a> • ⏬ <a href="https://huggingface.co/datasets/bluesky333/MedExQA" target="_blank">Dataset</a>  • ⚕️ <a href="https://huggingface.co/bluesky333/medphi2" target="_blank">MedPhi2</a><br>
</p>

## 🆕 News
- \[**July 2024**\] Our work is accepted at the ACL2024 BioNLP workshop. 
- \[**July 2024**\] We release <a href="https://huggingface.co/datasets/bluesky333/MedExQA" target="_blank">MedExQA</a> dataset. 
- \[**June 2024**\] We release <a href="https://huggingface.co/bluesky333/medphi2" target="_blank">MedPhi2</a> and <a href="https://arxiv.org/abs/2406.06331" target="_blank">Paper</a>. 

## Table of Contents
- [Benchmark Summary](#benchmark-summary)
- [Dataset Structure](#dataset-structure)
- [Dataset Structure](#data-fields)
- [Citation Information](#citation-information)
- [Leaderboard](#leaderboard)

##  Benchmark Summary
<a name="benchmark-summary"></a>
<a href="https://huggingface.co/datasets/bluesky333/MedExQA" target="_blank">MedExQA</a> is a novel benchmark in medical question-answering, to evaluate large language models’ (LLMs) understanding of medical knowledge through explanations.
The work is published in ACL2024 BioNLP workshop <a href="https://arxiv.org/abs/2406.06331" target="_blank">paper</a>. We address a major gap in current medical QA benchmarks which is the absence of comprehensive assessments of LLMs’ ability to generate nuanced medical explanations.
This work also proposes a new medical model, <a href="https://huggingface.co/bluesky333/medphi2" target="_blank">MedPhi2</a>, based on Phi-2 (2.7B). The model outperformed medical LLMs based on Llama2-70B in generating explanations.
From our understanding, it is the first medical language model benchmark with explanation pairs.

##  Dataset Structure
<a name="dataset-structure"></a>
Benchmark has 5 five distinct medical specialties that are underrepresented in current datasets.
| Specialty                   | # Examples | Explanation Similarity |
| -----                       | ---------- | ---------------------- |
| Biomedical Engineering      |    148     |  75.8 |
| Clinical Laboratory Science |      377   |  73.7 |
| Clinical Psychology         |      111   |  79.7 |
| Occupational Therapy        |   194      |  79.5 |
| Speech Language Pathology   |   135      |  80.5 |
| Total                       |   965      |  78.7 |
Explanation Similarity represents the average cosine similarity of the explanation pairs.

##  Data Fields
<a name="data-fields"></a>

The data files (tsv) are provided without headers.  
The fields are here explained as follows in the order of columns:  
* Column 1   : The question for the example  
* Column 2-5 : Choice A, B, C, D respectively  
* Column 6-7 : Explanation pairs  
* Column 8   : The right answer choice

##  Data Usage
<a name="data-usage"></a>

📝 installation
Follow the steps to configure the environment to use the code.

```shell  
git clone https://github.com/knowlab/MedExQA
cd MedExQA
conda create -n medexqa python=3.10 -y
conda activate medexqa
pip install -r requirements.txt
```
🔻Download the data
```
git clone https://huggingface.co/datasets/bluesky333/MedExQA
```

⚕️MedExQA Classification Using Logits
```shell  
python eval/evaluate_MedExQA.py -c bluesky333/medphi2 -d MedExQA -o cls_medphi2 --max-seq-len 2048 --batch-size 1
```
⚕️MedExQA Chat Classification
```shell  
python eval/evaluate_MedExQA.py -c bluesky333/medphi2 -d MedExQA -o chat_medphi2
```

##  Leaderboard
<a name="leaderboard"></a>

**Table: MCQ accuracy (%) using logits vs chat generation. 
The MCQ accuracy using logits is reported (except for GPT models). 
The performance gain/loss with chat generation approach is marked in parenthesis. 
"BE": Biomedical Engineering; "CP": Clinical Psychology; "SLP": Speech Language Pathology; 
"OT": Occupational Therapy; "CLS": Clinical Laboratory Science; "MAvg": Macro Average.**
| **Model**         | **BE**       | **CP**       | **SLP**       | **OT**       | **CLS**      | **MAvg**     |
|-------------------|--------------|--------------|---------------|--------------|--------------|--------------|
| Medinote-7B       | 33.6 (-4.9)  | 34.9 (-8.5)  | 23.1 (6.2)    | 38.1 (-8.5)  | 44.6 (-11.6) | 34.9 (-5.5)  |
| Meditron-7B       | 37.8 (-7.7)  | 46.2 (-16.0) | 20.8 (2.3)    | 42.9 (-10.6) | 43.3 (-6.7)  | 38.2 (-7.8)  |
| Llama2-7B         | 42.0 (-9.1)  | 47.2 (-9.4)  | 22.3 (1.5)    | 40.2 (-12.7) | 47.6 (-17.5) | 39.9 (-9.4)  |
| Asclepius-7B      | 44.8 (-11.2) | 47.2 (-17.0) | 27.7 (-1.5)   | 42.9 (-15.3) | 45.2 (-13.4) | 41.5 (-11.7) |
| Medinote-13B      | 46.2 (-18.9) | 52.8 (-30.2) | 28.5 (-4.6)   | 49.2 (-28.1) | 52.4 (-20.2) | 45.8 (-20.4) |
| AlpaCare-7B       | 53.2 (6.3)   | 53.8 (1.9)   | 26.9 (6.2)    | 59.8 (-3.7)  | 54.6 (-0.5)  | 49.6 (2.0)   |
| Asclepius-13B     | 57.3 (-21.0) | 56.6 (-33.0) | 25.4 (-3.8)   | 59.8 (-34.4) | 56.5 (-22.9) | 51.1 (-23.0) |
| Phi-2             | 61.5 (-35.7) | 68.9 (-38.7) | 26.2 (2.3)    | 64.0 (-43.4) | 50.0 (-25.0) | 54.1 (-28.1) |
| Llama2-13B        | 63.6 (-26.6) | 65.1 (-42.8) | 27.7 (16.2)   | 60.9 (-28.8) | 59.4 (-17.5) | 55.3 (-19.9) |
| MedPhi-2          | 65.7 (-5.6)  | 70.8 (0.0)   | 23.1 (0.0)    | 65.1 (-0.5)  | 55.1 (5.1)   | 56.0 (-0.2)  |
| AlpaCare-13B      | 67.1 (-4.9)  | 69.8 (-10.4) | 26.9 (-1.5)   | 65.1 (-4.8)  | 61.6 (-4.3)  | 58.1 (-5.2)  |
| Mistral           | 75.5 (-11.2) | 73.6 (-10.4) | 32.3 (-6.2)   | 75.7 (-6.3)  | 71.2 (0.0)   | 65.7 (-6.8)  |
| Meditron-70B      | 78.3 (-36.4) | 84.9 (-43.4) | 30.8 (-5.4)   | 69.8 (-37.0) | 68.6 (-24.2) | 66.5 (-29.3) |
| Yi                | 75.5 (-20.3) | 83.0 (-28.3) | 30.8 (0.8)    | 74.1 (-20.6) | 73.4 (-17.2) | 67.4 (-17.1) |
| SOLAR             | 74.8 (0.0)   | 81.1 (-2.8)  | **33.1** (-7.7)| 73.0 (-1.1) | 76.1 (-3.2)  | 67.6 (-3.0)  |
| InternLM2         | 77.6 (-25.2) | 82.1 (-38.7) | 29.2 (-5.4)   | 74.6 (-36.0) | 75.0 (-33.6) | 67.7 (-27.8) |
| ClinicalCamel     | 78.3 (-6.3)  | 84.0 (-14.1) | 28.5 (-5.4)   | 79.9 (-6.3)  | 75.8 (-6.2)  | 69.3 (-7.7)  |
| Llama2-70B        | 78.3 (-10.5) | 84.0 (-47.2) | 31.5 (-10.8)  | 80.4 (-44.4) | 72.9 (-29.8) | 69.4 (-28.5) |
| Med42             | 83.2 (-14.0) | 84.9 (-10.4) | 31.5 (-4.6)   | 79.4 (-13.8) | 80.9 (-12.6) | 72.0 (-11.1) |
| GPT3.5_1106       | 72.0         | 82.1         | 29.2          | 70.4         | 71.5         | 65.0         |
| GPT4_1106         | 86.7         | 86.8         | 31.5          | 88.4         | **91.7**     | 77.0         |
| GPT4_0125         | **90.2**     | **91.5**     | 30.8          | **90.0**     | **91.7**     | **78.8**     |

## Citation
<a name="citation-information"></a>
<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
```
@article{kim2024medexqa,
  title={MedExQA: Medical Question Answering Benchmark with Multiple Explanations},
  author={Kim, Yunsoo and Wu, Jinge and Abdulle, Yusuf and Wu, Honghan},
  journal={arXiv e-prints},
  pages={arXiv--2406},
  year={2024}
}
```