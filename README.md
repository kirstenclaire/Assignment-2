# Assignment-2
This API helps predict the probability of bankruptcy. 
- Uses XGBoost for predictions
- Dockerized for cloud deployment
- Hosted on Google Cloud Run
# Setup/Prep
Step 1: Clone Repository
Run the command below in your terminal:
```bash
git clone https://github.com/kirstenclaire/Assignment-2.git
cd Assignment-2
```
Step 2: Install Dependencies
```
pip install -r requirements.txt
```
# Run and Test API
```
uvicorn api.api:app --reload
```
---
| Method | Endpoint       | Description |
|--------|---------------|-------------|
| `GET`  | `/`           | Home Route Returns a welcome |
| `POST` | `/predict`    | Predict Bankruptcy using financial data |
---
```
curl -X GET "https://bankruptcy-api-107996658465.us-central1.run.app/"
```
```
http://127.0.0.1:8000/docs
```
# Deployment Instructions
# Build & Push Docker Image
```bash
docker build -t bankruptcy-api -f deployment/Dockerfile .
```
```
docker tag bankruptcy-api gcr.io/project-1-455004/bankruptcy-api
```
```
docker push gcr.io/project-1-455004/bankruptcy-api
```
# Deploy to Google Cloud Run
```
gcloud run deploy bankruptcy-api \
  --image gcr.io/project-1-455004/bankruptcy-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```
# Users
- Financial Analysts: To assess risk when making business decisions
- Loan Officer: Runs through API before making decisions regarding approvals

# Performance Metrics
To check performance metrics
```
python scripts/train.py
```
- F1 Score: 0.52
- Log Loss: 0.11
- AUC-ROC: 0.95
- Response Time: ~54ms
- Memory Usage: <512MB

