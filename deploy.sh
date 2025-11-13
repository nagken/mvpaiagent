#!/bin/bash

# MVP Google ADK Deployment Script for GCP
set -e

echo "Starting MVP Google ADK deployment to GCP..."

# Set project and region
PROJECT_ID="flawless-acre-401603"
REGION="us-central1"
SERVICE_NAME="mvp-google-adk"

echo "Setting up GCP configuration..."
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Enable required APIs
echo "Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Build and deploy to Cloud Run
echo "Building and deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "ENVIRONMENT=production" \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 80 \
    --timeout 3600 \
    --max-instances 10 \
    --min-instances 1

echo "Deployment completed!"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')"
echo "To view logs: gcloud logs tail --service=$SERVICE_NAME"
echo "To update: gcloud run deploy $SERVICE_NAME --source . --region $REGION"