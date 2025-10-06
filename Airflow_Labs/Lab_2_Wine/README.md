# 🍷 Wine Quality Prediction MLOps Pipeline

A comprehensive MLOps pipeline for predicting wine quality using ensemble machine learning models with Apache Airflow orchestration.

## 🎯 Project Overview

This project demonstrates a complete MLOps workflow that:

- **Predicts wine quality** (3-9 scale) using ensemble regression models
- **Uses real-world data** from UCI Wine Quality Dataset (local CSV files)
- **Handles both red and white wines** with comprehensive feature analysis
- **Automates the entire ML pipeline** from data loading to model evaluation
- **Provides production-ready monitoring** with Airflow UI and email notifications

## 🏗️ Architecture

### **Core Components**

1. **Apache Airflow DAG** (`dags/main.py`)

   - Orchestrates the entire ML pipeline
   - Manages task dependencies and execution
   - Provides monitoring and error handling

2. **Machine Learning Logic** (`dags/src/model_development.py`)

   - Data loading from local CSV files
   - Feature preprocessing and scaling
   - Ensemble model training (Random Forest, XGBoost, LightGBM, Linear Regression)
   - Model evaluation and performance metrics

3. **Data Storage** (`dags/data/`)
   - `winequality-red.csv` - Red wine dataset (1,599 samples)
   - `winequality-white.csv` - White wine dataset (4,898 samples)

## 📊 Dataset Information

### **Source**: UCI Machine Learning Repository (Local CSV Files)

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

### **Target**: Wine Quality (3-9 scale)

- **3-4**: Poor quality
- **5-6**: Average quality
- **7-8**: Good quality
- **9**: Excellent quality

## 🤖 Model Architecture

### **Ensemble Regression Models**

- **Random Forest**: Robust baseline with feature importance
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast gradient boosting with categorical support
- **Linear Regression**: Linear baseline model

### **Evaluation Metrics**

- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **Cross-validation**: Built-in model evaluation

## 🚀 Quick Start

### **Prerequisites**

- Python 3.11+
- Apache Airflow
- Required Python packages (see `requirements.txt`)

### **1. Install Dependencies**

```bash
cd /Users/hunjunsin/Desktop/Jun/MLOps/Airflow_Labs/Lab_2_Wine
pip install -r requirements.txt
```

### **2. Setup Airflow**

```bash
# Initialize Airflow database
airflow db migrate

# Start Airflow in standalone mode (creates admin user automatically)
airflow standalone
```

### **3. Access Airflow UI**

- **URL**: http://localhost:8080
- **Username**: `admin`
- **Password**: Check terminal output for generated password

### **4. Run the Pipeline**

1. Navigate to the Airflow UI
2. Find the "Wine_Quality_Prediction" DAG
3. Toggle the switch to enable the DAG
4. Click the "Play" button to trigger a manual run
5. Monitor the execution in real-time

## 📋 Pipeline Tasks

### **1. start_pipeline** (BashOperator)

- Displays pipeline start message
- Validates environment setup

### **2. load_wine_data** (PythonOperator)

- Loads red and white wine data from local CSV files
- Combines datasets and adds wine type encoding
- Saves processed data to pickle file

### **3. preprocess_wine_data** (PythonOperator)

- Separates features and target variable
- Splits data into train/test sets with stratification
- Applies StandardScaler for feature normalization
- Saves preprocessed data

### **4. build_save_wine_model** (PythonOperator)

- Trains ensemble of regression models
- Saves trained models to pickle file
- Performs cross-validation evaluation

### **5. load_evaluate_wine_model** (PythonOperator)

- Loads trained models and test data
- Evaluates each model's performance
- Displays comprehensive performance metrics

### **6. end_pipeline** (BashOperator)

- Displays pipeline completion message
- Triggers success email notification

## 🔧 Technical Implementation

### **Data Loading Strategy**

```python
# Local CSV file loading (no network dependency)
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
red_wine_path = os.path.join(data_dir, "winequality-red.csv")
white_wine_path = os.path.join(data_dir, "winequality-white.csv")
```

### **Ensemble Model Training**

```python
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
    'LinearRegression': LinearRegression()
}
```

### **Performance Evaluation**

```python
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"mse": mse, "r2": r2}
```

## 📈 Expected Performance

### **Typical Results**:

- **Random Forest R²**: 0.65-0.75
- **XGBoost R²**: 0.70-0.80
- **LightGBM R²**: 0.68-0.78
- **Linear Regression R²**: 0.60-0.70

### **Key Quality Factors**:

1. **Alcohol content** - Most important predictor
2. **Volatile acidity** - Strong negative correlation
3. **Sulphates** - Positive correlation with quality
4. **Citric acid** - Moderate positive correlation
5. **Density** - Negative correlation with quality

## 🚨 Monitoring & Alerts

### **Airflow UI Monitoring**

- **Real-time task status**: Green (success), Blue (running), Red (failed)
- **Detailed logs**: Click on any task to view execution logs
- **Task dependencies**: Visual graph showing task relationships
- **Performance metrics**: Execution time and resource usage

### **Email Notifications**

- **Success notifications**: Pipeline completion with performance summary
- **Failure alerts**: Detailed error messages and troubleshooting guidance
- **Configurable recipients**: Update email addresses in DAG configuration

## 🛠️ Customization Options

### **1. Different Datasets**

Replace CSV files in `dags/data/` directory:

```bash
# Add your own wine data
cp your_wine_data.csv dags/data/winequality-custom.csv
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

models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
    'LinearRegression': LinearRegression(),
    'SVR': SVR(),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 50))
}
```

## 🔮 Future Enhancements

### **1. Real-time Predictions**

- FastAPI endpoint for live quality scoring
- Batch prediction API for wine batches
- Model versioning and A/B testing

### **2. Advanced Analytics**

- Wine quality trend analysis over time
- Seasonal quality patterns
- Regional quality differences
- Feature importance visualization

### **3. Model Explainability**

- SHAP values for individual predictions
- Feature interaction analysis
- Quality factor importance ranking
- Model interpretability reports

### **4. Data Pipeline Enhancements**

- Automated data validation
- Data quality monitoring
- Feature drift detection
- Model performance monitoring

## 📚 Learning Outcomes

This project demonstrates:

- **MLOps Best Practices**: Production-ready pipeline with Airflow
- **Data Science**: Real-world dataset analysis and preprocessing
- **Machine Learning**: Ensemble regression techniques
- **Business Integration**: Wine industry applications
- **Monitoring**: Comprehensive pipeline monitoring and alerting
- **Scalability**: Modular design for easy extension

## 🐛 Troubleshooting

### **Common Issues**

1. **DAG not appearing in UI**

   - Check file permissions
   - Verify Python syntax
   - Check Airflow logs

2. **Task failures**

   - Check task logs in Airflow UI
   - Verify data file paths
   - Check Python dependencies

3. **Import errors**
   - Ensure all packages are installed
   - Check Python path configuration
   - Verify file structure

### **Debug Commands**

```bash
# Test DAG syntax
python dags/main.py

# Test individual functions
python -c "from dags.src.model_development import load_data; load_data()"

# Check Airflow configuration
airflow config list
```

## 🤝 Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**🍷 This MLOps pipeline showcases how to build production-ready machine learning systems for the wine industry, combining data science, machine learning, and DevOps best practices!** 🚀
