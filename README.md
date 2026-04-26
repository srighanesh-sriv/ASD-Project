<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d1117,100:1a3a5c&height=130&section=header&text=%F0%9F%A7%A0%20ASD%20Early%20Detection%20%E2%80%94%20Clinical%20ML%20System&fontSize=26&fontColor=58a6ff&animation=fadeIn" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-99%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/AUC-0.99-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Ethical%20Clearance-SRM%20Medical%20Council-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white"/>
</p>

# Detection of Autism Spectrum Disorder in Children — Machine Learning Pipeline

> Early detection of ASD in children under 2 years using PCA-enhanced machine learning. Deployed as a prototype web interface for real-time screening and therapy recommendation.

---

## Table of Contents

- [Context](#context)
- [Ethical Clearance](#ethical-clearance)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Context

Autism Spectrum Disorder (ASD) affects social interaction, communication, and behaviour. Current clinical diagnosis relies heavily on subjective behavioural assessment, often delayed until ages 3–4. This project addresses the gap: **can machine learning enable reliable early screening before age 2?**

The system analyses 21 features — spanning demographic data, medical history, and behavioural screening scores — to predict ASD likelihood. A positive prediction triggers **personalised therapy recommendations**, making the tool actionable for clinicians and parents alike.

---

## Ethical Clearance

This project was reviewed and approved by the **SRM Medical Council Ethics Committee**.

- Clearance valid for **3 years** from date of approval
- Data handling complies with institutional medical research standards
- Designed to assist, not replace, qualified clinical assessment

---

## Dataset

| Property | Detail |
|----------|--------|
| Domain | Autism Spectrum Disorder Screening (QCHAT) |
| Instances | 704 |
| Attributes | 21 |
| Target | `class_ASD` — 1 (ASD), 0 (Non-ASD) |
| Age range | Children under 24 months |

**Behavioural Screening Questions (QCHAT-10)**

Each question is scored 1 (yes) or 0 (no), totalled out of 10:

1. Abnormal speech development or language delay
2. Echolalia (immediate or delayed)
3. Limited interest in age-appropriate games
4. Odd, repetitive, or stereotyped behaviours
5. Unusual responses to sensory input (sounds, textures, tastes)
6. Little interest in peers
7. Does not point to desired objects
8. Separates easily from caregivers
9. Inappropriate laughing
10. Lacks awareness of danger

**Demographic and Medical Attributes:**

`Age_Mons` · `Sex` · `Ethnicity` · `Jaundice` · `Family_mem_with_ASD` · `Who_completed_test` · `Drinker` · `Chain_smoker`

---

## Methodology

### Pipeline Overview

```
Raw Data (XLSX)
    ↓
Preprocessing (Label Encoding, StandardScaler, OneHotEncoder)
    ↓
Dimensionality Reduction (PCA / t-SNE)
    ↓
Model Training + Cross-Validation
    ↓
Performance Evaluation (Accuracy, F1, Precision, Recall, ROC-AUC)
    ↓
Best Model Selection → Flask Web Interface
    ↓
Prediction + Therapy Recommendation Output
```

### Why PCA?

PCA was selected as the primary dimensionality reduction approach because:
- Captures maximum variance in fewer dimensions, reducing overfitting risk
- Improves Random Forest's generalisation on unseen data
- Speeds up training and inference on the 21-feature dataset
- Principal components are interpretable relative to the original features

### Models Evaluated

**Classical Models**

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Random Forest** | **99%** | **99** | **98** | **98.7** |
| SVM | 98.7% | 98.7 | 98 | 97 |
| KNN | 98% | 98 | 94 | 94.8 |
| Decision Tree | 97% | 97.3 | 94.7 | 94 |

**Proposed / Dimensionality Reduction Models**

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| **PCA + Random Forest** | **99%** | **98** | Selected model |
| t-SNE | 93.9% | 93.9 | Better for visualisation than classification |
| K-Means Clustering | 82.6% | 82.6 | Unsupervised baseline |

**Comparison with Prior Work**

| Model | Source | Accuracy |
|-------|--------|----------|
| PCA + Random Forest (this project) | — | **99%** |
| PCA + DNN | Mohanty et al., 2021 | 97% |
| K-Means Clustering | Rasul et al., 2023 | 86% |

---

## Results

- **Best model:** PCA + Random Forest — 99% accuracy, AUC = 0.99
- **Most predictive feature:** `family_mem_with_ASD_yes` (highest mutual information score)
- **Gender finding:** ASD prevalence significantly higher in males across all ethnic groups in dataset
- **Age finding:** Peak screening cases at 12 months; declines as age increases toward 24 months
- **QCHAT-10 Score** shows strong positive correlation (0.81) with ASD classification

---

## Installation

**Prerequisites:** Python 3.9+, pip

```bash
# Clone the repo
git clone https://github.com/srighanesh-sriv/asd-detection-ml.git
cd asd-detection-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web interface
python app.py
```

Open `http://localhost:8502` in your browser.

---

## Usage

**Via the web interface:**

1. Navigate to the **Prediction Page**
2. Enter the child's screening data:
   - Age in months, QCHAT-10 score
   - Sex, Ethnicity
   - Jaundice history, family ASD history
   - Who completed the screening test
3. Click **"Predict Recommendation"**
4. View ASD likelihood prediction and, if positive, a list of suggested therapies

**Therapy recommendations include** (based on prediction):
- Sensory Processing Therapy
- Animal-Assisted Therapy
- Early Intervention Programs
- Psychologist referral guidance

**Via Python directly:**

```python
import pickle
import pandas as pd

model = pickle.load(open('Autism.pkl', 'rb'))

# Sample input — adjust values for the child being screened
sample = pd.DataFrame([{
    'Age_Mons': 14,
    'Qchat_10_score': 7,
    'Sex': 'm',
    'Ethnicity': 'White European',
    'Jaundice': 'yes',
    'Family_mem_with_ASD': 'yes',
    'Who_completed_the_test': 'family member'
}])

prediction = model.predict(sample)
print("ASD Detected" if prediction[0] == 1 else "No ASD Detected")
```

---

## Project Structure

```
asd-detection-ml/
│
├── app.py                    # Flask web application
├── model.py                  # Training, evaluation, and model export
├── Autism.pkl                # Saved trained PCA + Random Forest model
├── requirements.txt
│
├── data/
│   └── autism_spectrum.xlsx  # Raw dataset
│
├── templates/
│   ├── index.html            # Home page
│   └── predict.html          # Prediction form and results
│
├── static/
│   └── style.css
│
├── screenshots/
│   ├── home.png
│   └── predict.png
│
└── README.md
```

---

## Limitations

- **Dataset size:** 704 instances limits generalisation across diverse demographics
- **Symptom heterogeneity:** ASD presents differently in children under 2, making consistent feature detection challenging
- **Ethical labelling:** Diagnosing very young children carries stigmatisation risk; tool is designed as a *screening aid*, not a diagnostic tool
- **Self-reported inputs:** QCHAT-10 responses depend on caregiver accuracy

---

## Future Work

- [ ] Integrate wearable sensor data for real-time behavioural signal collection
- [ ] Add multimodal inputs — speech patterns, eye tracking, facial expression analysis
- [ ] Longitudinal tracking to monitor developmental trajectories over time
- [ ] Incorporate genetic markers for enhanced prediction and subtype identification
- [ ] Expand dataset across more ethnic groups and socioeconomic backgrounds
- [ ] Clinical trial validation with partner institutions

---

## References

1. Frith & Happe (2005) — Autism Spectrum Disorder, *Current Biology*
2. Wall et al. (2012) — Use of AI to shorten behavioural diagnosis of autism, *Translational Psychiatry*
3. Thabtah (2017) — ASD Screening: ML adaptation and DSM-5 fulfillment
4. Hasan et al. (2022) — ML Framework for Early-Stage Detection of ASD
5. Mohanty, Parida & Patra (2021) — PCA with DNN model for ASD prediction
6. Rasul, Saha & Bala (2023) — K-Means clustering for ASD classification

---

## Author

**Srighanesh A S**  
B.Tech ECE — SRM Institute of Science and Technology (CGPA 9.35)  
Market Analyst @ Siemens, Chennai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Srighanesh-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/srighaneshsrivathsan)
[![Email](https://img.shields.io/badge/Email-srighanesh.sriv%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:srighanesh.sriv@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-srighanesh--sriv-181717?style=flat-square&logo=github)](https://github.com/srighanesh-sriv)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center"><i>Built with the goal of making early ASD intervention more accessible and evidence-based.</i></p>
