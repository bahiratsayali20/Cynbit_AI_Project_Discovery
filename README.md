
# Cynbit AI – Project Discovery Assistant

## About the Project

Cynbit AI is an AI/ML-based application that analyzes customer inquiries and predicts the software services required by customers.

## Key Features

- Customer inquiry classification
- AI-based service prediction
- Project requirement extraction
- Lead priority detection
- Suggested customer responses
- Interactive Streamlit dashboard
- Prediction history
- Downloadable CSV and text reports

## Service Categories

1. Web Development
2. AI Automation
3. App Development
4. Cloud Services

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Plotly

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
Cynbit-AI/
├── app.py
├── classifier.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── sample_inquiries.csv
```

## Machine Learning Model

The application uses TF-IDF vectorization and Logistic Regression for customer inquiry classification.

## Project Status

AI/ML proof of concept developed as part of the Cynbit Assessment Project.
