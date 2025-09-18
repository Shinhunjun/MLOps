# MLOps: Automated MNIST CNN Model Retraining Pipeline

A comprehensive MLOps pipeline that demonstrates how to maintain model performance in production by automatically retraining CNN models with real-world data using Streamlit, FastAPI, and GitHub Actions.

## 🎯 Project Overview

This project implements a complete MLOps workflow for a MNIST digit recognition system that:

- **Deploys** a CNN model via FastAPI for real-time predictions
- **Collects** user feedback through an intuitive Streamlit interface
- **Automatically retrains** the model when sufficient new data is available
- **Maintains** model performance without manual intervention

## 🏗️ MLOps Architecture

### Core Components

1. **FastAPI Backend** (`src/main.py`)

   - RESTful API for model inference
   - Real-time prediction endpoint (`/predict`)
   - Feedback collection endpoint (`/save-feedback`)
   - Automatic model reloading capabilities
   - Model versioning with timestamp-based filenames

2. **Streamlit Frontend** (`streamlit_app.py`)

   - User-friendly interface for image upload
   - Real-time prediction display
   - Feedback collection system
   - Model management controls

3. **GitHub Actions Workflow** (`.github/workflows/retrain.yml`)

   - Automated model retraining pipeline
   - Data validation and preprocessing
   - Model versioning and deployment
   - Continuous integration for ML models

4. **Automated Model Updater** (`auto_model_updater.py`)
   - Periodic model updates (every 10 minutes)
   - Git pull and model reloading
   - Background service for continuous improvement

## 🔄 MLOps Workflow Design

### 1. **Data Collection & Feedback Loop**

```
User Upload → Streamlit Interface → FastAPI Backend → Data Storage
     ↓
User Feedback → Validation → Training Data → Model Retraining
```

**Key Design Decisions:**

- **Subset-based data management**: Data is organized in `sub_set_N` folders to prevent Git conflicts
- **User validation**: Only confirmed feedback is used for training to ensure data quality
- **Batch processing**: Retraining triggers when 10 new data points are collected

### 2. **Automated Retraining Pipeline**

```
Data Threshold (10 samples) → GitHub Actions Trigger → Model Retraining
     ↓
Data Augmentation → Fine-tuning → Model Validation → Deployment
```

**MLOps Best Practices Implemented:**

- **Data augmentation**: Increases dataset diversity for better generalization
- **Fine-tuning approach**: Preserves learned features while adapting to new data
- **Model versioning**: Timestamp-based naming (`model_{timestamp}.h5`)
- **Rollback capability**: Maintains last 3 model versions for safety

### 3. **Continuous Deployment Strategy**

```
New Model → Git Push → Auto Pull → Model Reload → Live Service
```

**Deployment Strategy:**

- **Zero-downtime updates**: Model reloading without service interruption
- **Automatic synchronization**: Git pull on startup and periodic updates
- **Health monitoring**: Model status and performance tracking

## 🛠️ Technical Implementation

### Data Management Strategy

```python
# Subset-based data organization
new_data/
├── count.json          # Tracks current_count and sub_set_count
├── sub_set_0/          # First batch of 10 samples
├── sub_set_1/          # Second batch of 10 samples
└── sub_set_N/          # Nth batch of 10 samples
```

### Model Versioning System

```python
# Timestamp-based model naming
model_filename = f"cnn_mnist_model_{timestamp}.h5"
# Automatic cleanup of old models (keeps latest 3)
cleanup_old_models(timestamp)
```

### Automated Retraining Logic

```python
# Trigger condition
if current_count >= 10:
    # Increment sub_set_count
    # Create new sub_set_N folder
    # Trigger GitHub Actions workflow
    trigger_github_action()
```

## 🚀 Performance Maintenance Strategy

### 1. **Proactive Data Collection**

- **User feedback integration**: Every prediction includes user validation
- **Quality assurance**: Only confirmed correct predictions are used for training
- **Continuous learning**: Model adapts to real-world data patterns

### 2. **Automated Model Updates**

- **Threshold-based retraining**: Triggers when sufficient new data is available
- **Background processing**: Retraining happens without user intervention
- **Seamless deployment**: New models are automatically loaded into production

### 3. **Model Performance Monitoring**

- **Confidence scoring**: Tracks prediction confidence levels
- **User feedback analysis**: Monitors prediction accuracy trends
- **Model versioning**: Enables performance comparison across versions

## 📊 MLOps Benefits Achieved

### **Automation**

- ✅ **Zero manual intervention**: Complete end-to-end automation
- ✅ **Continuous improvement**: Model learns from every user interaction
- ✅ **Scalable architecture**: Handles increasing data volumes efficiently

### **Reliability**

- ✅ **Version control**: Git-based model and data versioning
- ✅ **Rollback capability**: Quick recovery from model degradation
- ✅ **Data integrity**: Subset-based organization prevents conflicts

### **Maintainability**

- ✅ **Modular design**: Clear separation of concerns
- ✅ **Monitoring**: Real-time model status and performance tracking
- ✅ **Documentation**: Comprehensive code documentation and workflows

## 🎯 Real-World Impact

This MLOps pipeline ensures that the MNIST digit recognition model:

1. **Adapts to new data patterns** without manual retraining
2. **Maintains high accuracy** as user behavior evolves
3. **Scales efficiently** with increasing user interactions
4. **Provides reliable predictions** through continuous model improvement

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- TensorFlow 2.x
- FastAPI
- Streamlit
- Git

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd MLOps

# Install dependencies
pip install -r FastAPI_Labs/requirements.txt

# Start all services
cd FastAPI_Labs
./start_servers.sh
```

### Manual Setup

```bash
# Terminal 1: Start FastAPI + Auto Updater
cd FastAPI_Labs
python auto_model_updater.py &

# Terminal 2: Start Streamlit
cd FastAPI_Labs
streamlit run streamlit_app.py --server.port 8501
```

## 📈 Future Enhancements

- **A/B testing framework** for model comparison
- **Advanced monitoring** with metrics dashboards
- **Distributed training** for larger datasets
- **Model explainability** features
- **Performance alerting** system

---

**This project demonstrates a production-ready MLOps pipeline that maintains model performance through automated retraining, ensuring reliable AI services in real-world applications.**
