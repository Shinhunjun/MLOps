# 🍷 Wine Quality Prediction Pipeline

An MLOps pipeline for predicting wine quality using the UCI Wine Quality Dataset with Apache Airflow orchestration.

## 🎯 Overview

This project implements a machine learning pipeline that:

- **Predicts wine quality** (0-10 scale) using ensemble regression models
- **Uses real-world data** from UCI Machine Learning Repository
- **Handles both red and white wines** with comprehensive feature analysis
- **Provides business insights** for wine production and quality control

## 📊 Dataset Information

### **Source**: UCI Machine Learning Repository

- **Red Wine**: 1,599 samples
- **White Wine**: 4,898 samples
- **Total**: 6,497 wine samples

### **Features** (12 chemical properties):

1. **Fixed acidity** - Non-volatile acids
2. **Volatile acidity** - Acetic acid content
3. **Citric acid** - Citric acid content
4. **Residual sugar** - Sugar remaining after fermentation
5. **Chlorides** - Salt content
6. **Free sulfur dioxide** - Free SO₂
7. **Total sulfur dioxide** - Total SO₂
8. **Density** - Wine density
9. **pH** - Acidity level
10. **Sulphates** - Potassium sulphate
11. **Alcohol** - Alcohol content (%)
12. **Wine type** - Red (1) or White (0)

### **Target**: Wine Quality (0-10 scale)

- **3**: Poor quality
- **4-6**: Average quality
- **7-8**: Good quality
- **9-10**: Excellent quality

## 🤖 Model Architecture

### **Ensemble Regression Model**

- **Random Forest**: Robust baseline with feature importance
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast gradient boosting with categorical support
- **Linear Regression**: Linear baseline model
- **Voting Regressor**: Soft voting for final predictions

### **Evaluation Metrics**

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: 5-fold CV for robust evaluation

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Setup Airflow**

```bash
# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

### 3. **Start Airflow Services**

```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler
airflow scheduler
```

### 4. **Access Airflow UI**

Open your browser and navigate to `http://localhost:8080`

## 📈 Business Applications

### **Wine Production**

- **Quality Control**: Predict quality before bottling
- **Process Optimization**: Identify key factors affecting quality
- **Cost Reduction**: Minimize quality issues early

### **Wine Industry**

- **Market Analysis**: Understand quality trends
- **Product Development**: Optimize wine characteristics
- **Competitive Advantage**: Data-driven quality improvement

## 🔧 Pipeline Features

### **1. Automated Data Loading**

- Direct download from UCI repository
- Fallback to synthetic data if download fails
- Data quality validation

### **2. Advanced Preprocessing**

- Feature scaling with StandardScaler
- Train/test split with stratification
- Comprehensive data validation

### **3. Ensemble Learning**

- Multiple algorithms for robust predictions
- Feature importance analysis
- Cross-validation for reliable performance

### **4. Comprehensive Evaluation**

- Multiple regression metrics
- Quality distribution analysis
- Feature importance ranking

## 📊 Expected Performance

### **Typical Results**:

- **R² Score**: 0.65-0.75
- **RMSE**: 0.6-0.8
- **MAE**: 0.4-0.6

### **Key Quality Factors**:

1. **Alcohol content** - Most important
2. **Volatile acidity** - Negative correlation
3. **Sulphates** - Positive correlation
4. **Citric acid** - Positive correlation
5. **Density** - Negative correlation

## 🛠️ Customization Options

### **1. Different Datasets**

```python
# Replace UCI URLs with your data source
red_wine_url = "your_red_wine_data.csv"
white_wine_url = "your_white_wine_data.csv"
```

### **2. Additional Features**

```python
# Add domain-specific features
df['alcohol_to_acid_ratio'] = df['alcohol'] / df['fixed acidity']
df['sugar_to_acid_ratio'] = df['residual sugar'] / df['fixed acidity']
```

### **3. Different Models**

```python
# Add more models to ensemble
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Add to VotingRegressor
('svr', SVR()),
('mlp', MLPRegressor(hidden_layer_sizes=(100, 50)))
```

## 📋 DAG Tasks

1. **load_wine_data**: Download wine dataset from UCI
2. **wine_data_quality_check**: Validate data quality
3. **preprocess_wine_data**: Scale features and split data
4. **separate_data_outputs**: Prepare data for training
5. **train_wine_quality_model**: Train ensemble model
6. **evaluate_wine_model**: Evaluate model performance
7. **wine_model_validation**: Validate model quality
8. **generate_wine_quality_report**: Generate analysis report

## 🚨 Monitoring & Alerts

### **Email Notifications**

- **Success**: Model training completion with performance metrics
- **Failure**: Error details and troubleshooting guidance

### **Airflow UI**

- Real-time task execution status
- Detailed logs and error messages
- Performance metrics visualization

## 🔮 Future Enhancements

### **1. Real-time Predictions**

- API endpoint for live quality scoring
- Batch prediction for wine batches

### **2. Advanced Analytics**

- Wine quality trend analysis
- Seasonal quality patterns
- Regional quality differences

### **3. Model Explainability**

- SHAP values for individual predictions
- Feature interaction analysis
- Quality factor importance

## 📚 Learning Outcomes

This project demonstrates:

- **Real-world MLOps**: Production-ready pipeline with Airflow
- **Data Science**: UCI dataset analysis and preprocessing
- **Machine Learning**: Ensemble regression techniques
- **Business Integration**: Wine industry applications

## 🤝 Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**🍷 This pipeline showcases how MLOps can be applied to the wine industry for quality prediction and process optimization!** 🚀
