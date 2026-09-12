# RETINOVA: Explainable AI Framework for Early Diabetic Retinopathy Detection

RETINOVA is an AI-assisted screening system for Diabetic Retinopathy (DR) that goes beyond a simple disease-grade prediction. It combines image preprocessing, deep learning-based classification, and Explainable AI (Grad-CAM) to produce a transparent, confidence-aware screening result from a retinal fundus photograph.

> **Final Year B.Tech Project** — CSE (Data Science), Shri Ramswaroop Memorial College of Engineering and Management, Lucknow (Affiliated to AKTU)
>
> **Team:** Akshay Shukla, Deepika Rai
> **Guide:** Dr. Sadhana Rana

---

## What it does

- **Preprocessing** — Enhances retinal images using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve lesion visibility.
- **Classification** — Predicts one of 5 DR severity grades (No DR, Mild, Moderate, Severe, Proliferative DR) using an EfficientNetV2 model.
- **Confidence & Referral** — Gives a confidence score and a simple Clear / Monitor / Refer recommendation.
- **Explainability** — Generates a Grad-CAM heatmap showing which regions of the retina influenced the model's prediction.

## Project Status

This repository currently contains a **working prototype** covering preprocessing, classification, and explainability. The following are planned for future phases (see [Future Scope](#future-scope)):

- [ ] Lesion detection & pixel-level segmentation (YOLOv11 + UNet++ on IDRiD dataset)
- [ ] Multi-backbone comparison (Swin Transformer, RETFound)
- [ ] Calibrated uncertainty estimation (ECE, Brier Score, Monte Carlo Dropout)
- [ ] Separate doctor and patient dashboards
- [ ] Automated PDF report generation
- [ ] Clinical metadata fusion (age, HbA1c, blood pressure, etc.)

## Repository Structure

```
RETINOVA/
├── app.py                          # Streamlit demo application
├── requirements.txt                # Python dependencies
├── RETINOVA_starter.ipynb          # Colab notebook: training pipeline
├── retinova_efficientnetv2.pth     # Trained model weights (not tracked in git — see below)
└── README.md
```

> **Note:** Model weights (`.pth`) and datasets are excluded from this repository via `.gitignore` since they are too large for GitHub. See the setup instructions below to regenerate them.

## Datasets Used

| Dataset | Purpose | Link |
|---|---|---|
| APTOS 2019 Blindness Detection | DR severity classification | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data) |
| IDRiD | Lesion segmentation (planned) | [Grand Challenge](https://idrid.grand-challenge.org/) |
| EyePACS (Diabetic Retinopathy Detection) | Robustness testing (planned) | [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) |
| Messidor-2 | External validation (planned) | [ADCIS](https://www.adcis.net/en/third-party/messidor2/) |

## How to Run

### 1. Train the model (Google Colab)

Open `RETINOVA_starter.ipynb` in Google Colab, set the runtime to a GPU (T4), and run all cells in order. You'll need a free [Kaggle API token](https://www.kaggle.com/settings) to download the dataset. Training produces `retinova_efficientnetv2.pth`.

### 2. Run the demo app locally

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Place the trained retinova_efficientnetv2.pth in this folder, then run:
streamlit run app.py
```

The app opens in your browser. Upload a fundus image to see the predicted DR grade, confidence score, referral recommendation, and Grad-CAM explanation.

## Tech Stack

- **Deep Learning:** PyTorch, timm (EfficientNetV2)
- **Explainability:** pytorch-grad-cam
- **Image Processing:** OpenCV, Albumentations
- **App/Demo:** Streamlit

## Applications

- Community and rural DR screening camps
- Tele-ophthalmology second-opinion support
- Diabetic patient self-monitoring
- Clinic decision-support pre-triage

## Limitations

This system is a research/academic prototype and is **not intended for clinical diagnosis**. It has not been validated on diverse real-world populations, does not currently perform lesion-level segmentation, and does not meet healthcare data privacy or regulatory requirements for clinical deployment.

## Future Scope

- **Explanation-Fidelity Score** — a quantitative IoU/Dice-based metric comparing Grad-CAM heatmaps against UNet++ ground-truth lesion masks.
- **Calibrated Uncertainty-Aware Referral** — mapping calibrated confidence directly to referral decisions.
- External validation on held-out datasets (DDR, Messidor).
- Multilingual patient-facing reports.

## References

Key literature this project builds on includes Gulshan et al. (2016) on deep learning for DR detection, Abramoff et al. (2018) on the IDx-DR clinical validation, Selvaraju et al. (2017) on Grad-CAM, and Zhou et al. (2018) on UNet++. Full reference list available in the project synopsis document.

---

*This project is developed for academic purposes as part of the B.Tech curriculum at SRMCEM, Lucknow, affiliated to Dr. APJ Abdul Kalam Technical University.*
