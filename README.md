# Support Ticket Classification for VoltMart Electronics

VoltMart Electronics is an online store that receives dozens of customer support tickets every day.
Agents spend time reading each ticket and sending it to the right department. This project builds a
small machine-learning classifier that suggests the correct category automatically, so staff can
respond faster.

This repository covers **Task 1: set up the repository and load sample support tickets**.
Later tasks will add a baseline model, an improved model, and a FastAPI service.

## What is in this repository

| File | Purpose |
|------|---------|
| `load_and_clean_tickets.ipynb` | Downloads the dataset, cleans it, and saves `cleaned_tickets.csv` |
| `cleaned_tickets.csv` | The cleaned dataset, ready for modeling |
| `requirements.txt` | Python packages needed to run the notebook |
| `README.md` | This file |

## Dataset

A public customer-support-ticket dataset from Hugging Face
(`interneuronai/customer_support_ticket_classification_pegasus_dataset`). The notebook downloads it
automatically, so you do not need an account or an API key.

## Cleaning steps

1. Tidy column names (lower-case, underscores instead of spaces).
2. Drop unused columns (empty columns and leftover index columns).
3. Lower-case all text.
4. Remove rows with missing values (NaN) and duplicate rows.
5. Save the result as `cleaned_tickets.csv`.

The notebook shows the first five rows of the raw data and of the cleaned data.

## Setup instructions

You need Python 3.9 or newer and Git.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/voltmart-ticket-classifier.git
cd voltmart-ticket-classifier

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# 3. Install the packages
pip install -r requirements.txt

# 4. Start Jupyter and open the notebook
jupyter notebook load_and_clean_tickets.ipynb
```

In Jupyter choose **Kernel -> Restart & Run All**. When it finishes, `cleaned_tickets.csv` is
created in the project folder.

## Next steps

- Task 2: build a baseline classifier and beat it with cross-validation.
- Task 3: wrap the model in a FastAPI service and publish it on Hugging Face Spaces.
