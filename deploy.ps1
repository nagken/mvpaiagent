# MVP Google ADK Deployment Script for GCP (PowerShell)

Write-Host "🚀 Starting MVP Google ADK deployment to GCP..." -ForegroundColor Green

# Set project and region
$PROJECT_ID = "flawless-acre-401603"
$REGION = "us-central1"
$SERVICE_NAME = "mvp-google-adk"

Write-Host "📝 Setting up GCP configuration..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Enable required APIs
Write-Host "🔧 Enabling required GCP APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Build and deploy to Cloud Run
Write-Host "🏗️ Building and deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME `
    --source . `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --set-env-vars "ENVIRONMENT=production" `
    --memory 2Gi `
    --cpu 2 `
    --concurrency 80 `
    --timeout 3600 `
    --max-instances 10 `
    --min-instances 1

Write-Host "✅ Deployment completed!" -ForegroundColor Green

# Get service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"
Write-Host "🌐 Service URL: $SERVICE_URL" -ForegroundColor Cyan
Write-Host "📊 To view logs: gcloud logs tail --service=$SERVICE_NAME" -ForegroundColor Blue
Write-Host "🔧 To update: gcloud run deploy $SERVICE_NAME --source . --region $REGION" -ForegroundColor Blue