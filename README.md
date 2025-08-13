# Job Recommendation System

This project provides a job recommendation system using TF-IDF vectorization and an XGBoost regression model to predict job relevance scores for search queries.

## Project Structure

```
Job_recommendation_system/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── model.ipynb                # Model training notebook
├── clean_text.py              # Text preprocessing utilities
├── arbitrary_query.py         # Interactive query interface
├── ad_detail_v1.jsonl         # Job advertisement data
├── qry_rel_v1.jsonl          # Query-relevance training data
├── model/                     # Saved model artifacts
│   ├── tfidf_vectorizer.pkl   # Trained TF-IDF vectorizer
│   ├── xgboost_model.pkl      # Trained XGBoost model
│   └── job_data.pkl           # Preprocessed job data
```

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install OpenMP (required for XGBoost)
- **Windows:**  
  XGBoost wheels include OpenMP support.
- **Linux:**  
  ```bash
  sudo apt-get install libgomp1
  ```
- **Mac:**  
  ```bash
  brew install libomp
  ```

## Usage

### Option 1: Use Pre-trained Model 
Run the interactive query tool with saved model:
```bash
python arbitrary_query.py
```
Enter search queries to get top 5 job recommendations with relevance scores.

### Option 2: Train Model from Scratch
Open and run `model.ipynb` to:
- Load and preprocess job ads and query data
- Train TF-IDF vectorizer and XGBoost model
- Evaluate model performance
- Save trained artifacts to `model/` directory
## Files
- `model.ipynb` - Model training notebook
- `clean_text.py` - Text preprocessing utilities
- `arbitrary_query.py` - Interactive query interface
- `ad_detail_v1.jsonl` - Job advertisement data
- `qry_rel_v1.jsonl` - Query-relevance data
- `model/` - Saved model artifacts
