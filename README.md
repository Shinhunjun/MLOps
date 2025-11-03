# MLOps Course Assignments

A collection of MLOps projects demonstrating end-to-end machine learning pipelines, automated retraining workflows, and cloud deployment strategies.

## 📚 Projects Overview

This repository contains three comprehensive MLOps assignments, each showcasing different aspects of production machine learning systems:

### 1. 🍷 [Airflow_Labs - Wine Quality Prediction Pipeline](./Airflow_Labs/Lab_2_Wine/)

**Apache Airflow orchestration for ML workflows**

A complete MLOps pipeline for predicting wine quality using ensemble machine learning models with Apache Airflow orchestration.

**Key Features:**
- **Ensemble Models**: Random Forest, XGBoost, LightGBM, and Linear Regression
- **Automated Pipeline**: Apache Airflow DAG orchestrating the entire ML workflow
- **Data Processing**: UCI Wine Quality Dataset with comprehensive feature analysis
- **Production Monitoring**: Airflow UI with task dependencies and email notifications
- **Error Handling**: Robust retry mechanisms and failure notifications

**Tech Stack:** Apache Airflow, Scikit-learn, XGBoost, LightGBM, Pandas

**[→ View Wine Quality Pipeline Details](./Airflow_Labs/Lab_2_Wine/README.md)**

---

### 2. 🔄 [FastAPI_Streamlit_GithubAciton_Labs - MNIST Automated Retraining](./FastAPI_Streamlit_GithubAciton_Labs/)

**Continuous learning with automated model retraining**

An MLOps system demonstrating continuous model improvement through user feedback and automated retraining using GitHub Actions.

**Key Features:**
- **Real-time API**: FastAPI backend for MNIST digit recognition
- **User Feedback Loop**: Streamlit interface for data collection and validation
- **Automated Retraining**: GitHub Actions triggers retraining when 10 new samples collected
- **Zero-downtime Deployment**: Automatic model updates without service interruption
- **Data Management**: Subset-based organization preventing Git conflicts
- **Model Versioning**: Timestamp-based naming with automatic cleanup

**Tech Stack:** FastAPI, Streamlit, GitHub Actions, TensorFlow, Git automation

**Workflow:**
```
User Upload → Feedback Collection → Data Threshold (10 samples)
     ↓
GitHub Actions → Model Retraining → Git Push
     ↓
Auto Pull → Model Reload → Live Service Update
```

---

### 3. ☁️ [GithubAction_GCP_Docker - Cloud-Native MLOps Pipeline](./GithubAction_GCP_Docker/)

**Production deployment on Google Cloud Platform**

A complete cloud-native MLOps pipeline with automated retraining, deployed on GCP with modern frontend and backend architecture.

**Live Demo:**
- Frontend: https://frontend-mh7kjdw4p-shinhunjuns-projects.vercel.app
- Backend API: https://mnist-api-762303020827.us-central1.run.app

**Key Features:**
- **Cloud Deployment**: Backend on Cloud Run, Frontend on Vercel
- **Vertex AI Integration**: Model registry with versioning
- **Cloud Storage**: GCS for feedback data and training datasets
- **GitHub Actions CI/CD**: Automated retraining and deployment pipeline
- **Modern Frontend**: React with interactive canvas for digit drawing
- **Containerization**: Docker for reproducible deployments
- **Scalable Architecture**: Serverless infrastructure with auto-scaling

**Tech Stack:**
- **Backend**: FastAPI, TensorFlow, Google Cloud Run, Docker
- **Frontend**: React.js, Vercel
- **MLOps**: Vertex AI Model Registry, GCS, GitHub Actions
- **Infrastructure**: Google Cloud Platform, Container Registry

**Architecture:**
```
React Frontend (Vercel) → FastAPI Backend (Cloud Run)
                              ↓
                    Vertex AI Model Registry
                              ↓
                    Feedback → GCS Storage
                              ↓
                    GitHub Actions Retraining
                              ↓
                    New Model → Vertex AI → Live Deployment
```

**[→ View Cloud-Native Pipeline Details](./GithubAction_GCP_Docker/README.md)**

---

## 🎯 Learning Outcomes

### Assignment 1: Airflow Orchestration
- Pipeline orchestration with Apache Airflow
- DAG design and task dependency management
- Ensemble model development
- Production monitoring and alerting

### Assignment 2: Automated Retraining
- Continuous learning from user feedback
- GitHub Actions for ML workflows
- Zero-downtime model updates
- Data versioning strategies
- API design for ML services

### Assignment 3: Cloud-Native MLOps
- Cloud deployment (GCP, Vercel)
- Container orchestration with Docker
- Serverless architecture
- Model registry management (Vertex AI)
- Full-stack ML application development
- Cloud storage integration
- CI/CD for machine learning

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Docker (for Assignment 3)
- Apache Airflow (for Assignment 1)
- Google Cloud Platform account (for Assignment 3)
- GitHub account (for Assignments 2 & 3)

### Running Projects Locally

**Assignment 1: Airflow Wine Quality Pipeline**
```bash
cd Airflow_Labs/Lab_2_Wine
./setup.sh
# Access Airflow UI at http://localhost:8080
```

**Assignment 2: FastAPI + Streamlit Automated Retraining**
```bash
cd FastAPI_Streamlit_GithubAciton_Labs
pip install -r requirements.txt
./start_servers.sh
# Access Streamlit at http://localhost:8501
# API docs at http://localhost:8000/docs
```

**Assignment 3: GCP Cloud Deployment**
```bash
cd GithubAction_GCP_Docker
# See detailed setup instructions in the folder README
```

---

## 📊 Technology Comparison

| Feature | Assignment 1 | Assignment 2 | Assignment 3 |
|---------|-------------|-------------|-------------|
| **Orchestration** | Apache Airflow | GitHub Actions | GitHub Actions |
| **API Framework** | N/A | FastAPI | FastAPI |
| **Frontend** | N/A | Streamlit | React.js |
| **Deployment** | Local | Local/VPS | Cloud (GCP + Vercel) |
| **Model Storage** | Local files | Git repository | Vertex AI Registry |
| **Data Storage** | Local CSV | Git + local files | Google Cloud Storage |
| **Containerization** | No | No | Docker |
| **Auto-scaling** | No | No | Yes (Cloud Run) |
| **Model Versioning** | Manual | Timestamp-based | Vertex AI managed |

---

## 🏆 Best Practices Demonstrated

### MLOps Principles
- ✅ **Automated pipelines** for reproducibility
- ✅ **Version control** for models and data
- ✅ **Continuous integration/deployment** for ML
- ✅ **Monitoring and logging** for production systems
- ✅ **Scalable architecture** for growing datasets

### Software Engineering
- ✅ **Modular code design** for maintainability
- ✅ **API-first architecture** for flexibility
- ✅ **Error handling and retries** for reliability
- ✅ **Documentation** for collaboration
- ✅ **Testing strategies** for quality assurance

### Cloud-Native Development
- ✅ **Containerization** for consistency
- ✅ **Serverless deployment** for cost efficiency
- ✅ **Cloud storage** for scalability
- ✅ **Managed services** for reduced operational overhead

---

## 📖 Additional Resources

Each project folder contains:
- Detailed README with setup instructions
- Architecture diagrams and workflow explanations
- Code documentation and comments
- Troubleshooting guides

---

## 🔗 Repository Structure

```
MLOps/
├── Airflow_Labs/
│   └── Lab_2_Wine/              # Assignment 1: Airflow orchestration
│       ├── dags/                # Airflow DAGs and ML code
│       └── README.md            # Detailed documentation
│
├── FastAPI_Streamlit_GithubAciton_Labs/  # Assignment 2: Automated retraining
│   ├── src/                     # FastAPI backend
│   ├── streamlit_app.py         # Frontend interface
│   ├── new_data/                # Collected feedback data
│   └── README.md                # Empty (see root README)
│
├── GithubAction_GCP_Docker/     # Assignment 3: Cloud-native MLOps
│   ├── backend/                 # FastAPI application
│   ├── frontend/                # React application
│   ├── scripts/                 # Training and deployment scripts
│   ├── .github/workflows/       # CI/CD pipelines
│   └── README.md                # Comprehensive cloud deployment guide
│
└── README.md                    # This file
```

---

**These projects demonstrate production-ready MLOps practices for building, deploying, and maintaining machine learning systems at scale.**
