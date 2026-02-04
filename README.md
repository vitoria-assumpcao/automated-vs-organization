#  Automated Vestibular Schwannoma Dataset Organization

**Automated organization pipeline for Vestibular Schwannoma MRI data (VS-SEG Dataset).**
This repository provides a fully automated workflow for converting, organizing, and preparing MRI data from the *Vestibular Schwannoma SEG (VS-SEG)* dataset for deep learning applications, such as tumor segmentation.

---

##  Overview

This project automates the preprocessing of medical images for vestibular schwannoma segmentation.
Raw DICOM images and expert contour annotations are standardized into structured datasets suitable for training segmentation models (e.g., U-Net, YOLO-based pipelines, or hybrid detection–segmentation approaches).

## 📊 Data Source

The MRI data used in this project come from the **Vestibular Schwannoma SEG (VS-SEG)**
dataset, publicly available through **The Cancer Imaging Archive (TCIA)**.

TCIA is part of the **NCI Cancer Research Data Commons (CRDC)** and can be accessed
via the **Imaging Data Commons (IDC)** platform, which provides cloud-based tools
for data exploration and analysis.


## Main features

* Organizes DICOM and contour files by patient (`VS-SEG-xxx`) and MRI sequence
* Supports preprocessing of **T1-weighted** MRI sequences in both **tumor-only** and **full-exam** modes
* Converts DICOM → PNG (8-bit)
* Converts tumor contours (LPS → pixel coordinates, `.txt`)
* Generates binary tumor masks
* Splits the dataset into `train`, `valid`, and `test` sets


---

## ⚙️ Environment setup

### Install dependencies 
This project uses Poetry for dependency management.

```bash
poetry install
```
This command installs all required libraries and creates the virtual environment.

---

## 📂 Dataset Preparation Workflow

### Step 1 — Organize source data

Place the original VS-SEG data in the following structure:

```
/source_data/
 ├── Vestibular-Schwannoma-SEG-contours-Mar-2021/contours/
 └── manifest-1614264588831/
```
Paths are configured via the .env file.

---
### Step 2 — Running the pipeline
From the root of the repository, run:
```bash
 python -m automated_vs_organization.pipeline
```
The pipeline:

1. Organizes raw DICOM images
2. Organizes contour annotations
3. Generates images, labels, and masks
4. Builds the final dataset splits

The pipeline can be configured (inside pipeline.py) to generate:

* tumor-only datasets, or
* full datasets (including slices without tumor).
---
### Step 3 — Output Structure

After execution, the pipeline produces:

**Organized data**
```
/DICOM_Organized/VS-SEG-XXX/
 ├── t1
 └── t2
/Contours_Organized/VS-SEG-XXX/
 ├── t1_contours.json
 └── t2_contours.json
```

**Preprocessed dataset**
```
/dataset_vs_tumor/ or dataset_vs_full/
 ├── images/
 ├── masks/
 └── labels/
```
**Training-ready dataset**
```
/training_set_vs_*/
 ├── train/
 ├── val/
 ├── test/
 └── splits.json
```
All outputs are fully aligned (1:1 correspondence between images, masks, and labels).
Individual scripts can also be executed independently, if required.

---
## Checking and Validation Scripts
This repository also includes validation scripts to verify dataset consistency after the pipeline execution.
The alignment check script ensures that images, labels, and masks are correctly matched:
 ```bash
 cd check_scripts/
 python check_alignment.py
```
In addition, a structure name inspection script is provided to analyze how tumor and non-tumor structures are named across all contour files:
 ```bash
 cd check_scripts/
 python structure_names.py
```
This script helps identify naming variations in the original annotations and supports the definition of robust structure name matching rules during preprocessing.

---

## 🧾 Citation

- The Cancer Imaging Archive (TCIA): Vestibular Schwannoma-SEG Dataset

> Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I., Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm (version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.9YTJ-5Q73 

---

## 👩‍💻 Author

**Vitória Assumpção**
Biomedical Informatics – UFCSPA
Research in Medical Image Segmentation and AI for Healthcare.

---

