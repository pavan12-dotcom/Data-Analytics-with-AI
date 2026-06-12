#!/bin/bash

# ==============================================================================
# Cloud Deployment Script: AWS, Google Cloud (GCP), and Microsoft Azure
# Project: AI-Driven Customer Behavior Analytics Platform
# ==============================================================================

IMAGE_NAME="customer-analytics"
TAG="latest"

echo "======================================================================"
echo "Starting deployment setup for: $IMAGE_NAME:$TAG"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. GOOGLE CLOUD PLATFORM (GCP) - Google Cloud Run (Recommended & Simplest)
# ------------------------------------------------------------------------------
deploy_gcp() {
    PROJECT_ID=$1
    REGION="us-central1"
    
    if [ -z "$PROJECT_ID" ]; then
        echo "Error: GCP Project ID is required for GCP deployment."
        exit 1
    fi
    
    echo "Deploying to Google Cloud Run in project: $PROJECT_ID..."
    
    # Enable Cloud Run & Container Registry APIs
    gcloud services enable run.googleapis.com containerregistry.googleapis.com --project="$PROJECT_ID"
    
    # Configure Docker local credentials to talk to GCP
    gcloud auth configure-docker --quiet
    
    # Tag and Push the local docker image to Google Container Registry
    docker tag "$IMAGE_NAME:$TAG" "gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG"
    docker push "gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG"
    
    # Deploy the container to Cloud Run
    gcloud run deploy "$IMAGE_NAME" \
        --image "gcr.io/$PROJECT_ID/$IMAGE_NAME:$TAG" \
        --region "$REGION" \
        --platform managed \
        --allow-unauthenticated \
        --port 8501 \
        --project "$PROJECT_ID"
        
    echo "GCP Deployment Complete!"
}

# ------------------------------------------------------------------------------
# 2. AMAZON WEB SERVICES (AWS) - AWS App Runner or AWS ECS Fargate
# ------------------------------------------------------------------------------
deploy_aws() {
    AWS_ACCOUNT_ID=$1
    REGION="us-east-1"
    
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo "Error: AWS Account ID is required for AWS deployment."
        exit 1
    fi
    
    echo "Deploying to AWS Elastic Container Registry (ECR)..."
    
    # Login to ECR
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
    
    # Create the repository if it doesn't exist
    aws ecr create-repository --repository-name "$IMAGE_NAME" --region "$REGION" || true
    
    # Tag and push image
    docker tag "$IMAGE_NAME:$TAG" "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:$TAG"
    docker push "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:$TAG"
    
    echo "AWS Push Complete!"
    echo "To deploy to AWS ECS Fargate:"
    echo "  1. Create a task definition using the ECR image URI: $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:$TAG"
    echo "  2. Launch an ECS Service under Fargate with port mapping 8501."
}

# ------------------------------------------------------------------------------
# 3. MICROSOFT AZURE - Azure Container Instances (ACI)
# ------------------------------------------------------------------------------
deploy_azure() {
    RESOURCE_GROUP=$1
    CONTAINER_NAME="customer-analytics-app"
    
    if [ -z "$RESOURCE_GROUP" ]; then
        echo "Error: Azure Resource Group name is required for Azure deployment."
        exit 1
    fi
    
    echo "Deploying to Azure Container Instances..."
    
    # Build the image and push to Azure Container Registry (ACR) if available
    # Or deploy the image directly to Azure Container Instances from Docker Hub/Public registry
    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --image "$IMAGE_NAME:$TAG" \
        --dns-name-label "customer-analytics-dashboard" \
        --ports 8501 \
        --cpu 1 \
        --memory 1.5 \
        --location eastus
        
    echo "Azure Deployment Complete!"
}

# Helper usage instructions
usage() {
    echo "Usage: ./deploy.sh [gcp|aws|azure] [arguments]"
    echo "  ./deploy.sh gcp <gcp-project-id>"
    echo "  ./deploy.sh aws <aws-account-id>"
    echo "  ./deploy.sh azure <azure-resource-group>"
}

# Command dispatcher
case "$1" in
    gcp)
        deploy_gcp "$2"
        ;;
    aws)
        deploy_aws "$2"
        ;;
    azure)
        deploy_azure "$2"
        ;;
    *)
        usage
        ;;
esac
