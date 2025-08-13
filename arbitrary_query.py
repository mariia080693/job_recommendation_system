import joblib
import pandas as pd
import time
from xgboost import XGBRegressor
from scipy.sparse import hstack, vstack
from clean_text import clean_text

# Load model artifacts
MODEL_DIR = "model"
tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
tfidf_job_matrix = joblib.load(f"{MODEL_DIR}/tfidf_job_matrix.joblib")
job_ads_df = pd.read_json("ad_detail_v1.jsonl", lines=True)

# Load XGBRegressor
xgb_model = XGBRegressor()
xgb_model.load_model(f"{MODEL_DIR}/xgb_model.json")

print("Model loaded successfully!")

while True:
    user_query = input("Enter job search query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    
    start_time = time.time()
    clean_query = clean_text(user_query)
    query_vec = tfidf_vectorizer.transform([clean_query])
    if query_vec.nnz == 0:
        print("Error: No job matches; all job relevances are 0 for this query.")
        continue
    query_matrix = vstack([query_vec] * tfidf_job_matrix.shape[0])
    pair_matrix = hstack([tfidf_job_matrix, query_matrix])
    
    # Make predictions
    prediction_scores = xgb_model.predict(pair_matrix)
    
    # Get top 5 job recommendations
    top_indices = prediction_scores.argsort()[::-1][:5]
    results = job_ads_df.loc[top_indices][['ad_id', 'title']]
    results['score'] = prediction_scores[top_indices]

    elapsed_time = time.time() - start_time

    print(f"\nTop recommendations for '{user_query}':")
    print(results)
    print(f"\nInference time: {elapsed_time:.3f}s")