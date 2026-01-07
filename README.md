# Data Collection, Cleaning & Exploratory Data Analysis (EDA)
“EDA is the process of exploring a dataset to understand its structure, quality, patterns, outliers, and relationships, so we can make correct decisions before modeling or reporting.”
## Project Overview
This project demonstrates a complete beginner-friendly workflow in Python (Pandas) for:
1) Collecting a dataset (CSV)
2) Cleaning the dataset (handling missing values, duplicates, formatting issues)
3) Performing Exploratory Data Analysis (EDA) to understand patterns, outliers, and relationships
4) Building simple visualizations using Matplotlib and Seaborn

The goal is to practice real-world data preprocessing and basic analysis skills.

## Dataset
This project uses the **Titanic dataset** (public sample dataset) for learning purposes.

## What I Did (Step-by-step)

### 1) Data Loading (Collection)
Loaded the CSV dataset into a Pandas DataFrame.

### 2) Data Cleaning
Checked dataset shape, columns, and data types
Removed duplicate rows
Standardized text formatting (trim spaces, consistent casing)
Identified missing values (count + %)
Dropped columns with very high missing values (e.g., `Cabin`)
Filled missing values:
  - Numeric columns → **median**
  - Categorical columns → **mode** (most frequent value)
Exported the cleaned dataset to a new CSV file

### 3) Exploratory Data Analysis (EDA)
- Computed basic statistics:
  - mean, median, min, max
- Analyzed categorical distributions using value counts
- Detected outliers using the **IQR method**
- Checked correlations among numeric variables
- Visualized insights using:
  - Histograms (distributions)
  - Bar charts (category counts and survival rates)
  - Line plot (trend across age groups)

## Files in This Repository
- `Project1_Cleaning_EDA.ipynb`  
  Notebook containing full cleaning + EDA workflow

- `project1_data_cleaning.py`  
  Script version of the cleaning process

  - `project1_eda.py`  
  Script version of the EDA process

- `titanic_cleaned.csv`  
  Output cleaned dataset (optional)


## How to Run
### Option 1 (Recommended): Jupyter Notebook
1. Open `Project1_Cleaning_EDA.ipynb` in Jupyter Notebook / Google Colab
2. Run cells from top to bottom

### Option 2: Python Script
### Requirements
Install dependencies:
pip install pandas numpy matplotlib seaborn
### Check results in:
titanic_cleaned.csv
eda_outputs/

```bash
# Run For Cleaning of data
python project1_data_cleaning.py
# Run for EDA
python project2_eda.py




