# Cynbit AIML Projects

This workspace contains several Python-based learning projects developed during a summer internship at Cynbit Technologies.

## Student Performance Predictor

A machine learning application that predicts student performance categories (`High`, `Medium`, `Low`) based on attendance, study habits, previous test scores, and other academic features.

### Run the predictor app

```bash
cd StudentPerformancePrediction
python3 -m streamlit run app.py
```

### Notes

- Upload your own student dataset CSV or use the provided sample data.
- The app trains a classification model and displays accuracy, classification report, confusion matrix, and prediction confidence.
- The sample dataset generation script is located at `StudentPerformancePridector/generate_data.py`.