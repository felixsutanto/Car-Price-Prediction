from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import datetime
import json
import base64
from io import BytesIO
from car_price_model import CarPriceModel

app = Flask(__name__)

class FlaskDeployment:
    def __init__(self):
        """Initialize Flask deployment interface"""
        # Initialize model
        self.model = CarPriceModel()
        
        # Make sure the models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Try to load existing model, or build if not available
        model_path = 'models/car_price_model.pkl'
        if os.path.exists(model_path):
            try:
                self.model.model = joblib.load(model_path)
                self.model.load_and_preprocess_data()
                self.model.analyze_depreciation()
                print("Loaded existing model")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.load_and_build_model()
        else:
            self.load_and_build_model()
    
    def load_and_build_model(self):
        """Load data and build model"""
        print("Loading data...")
        self.model.load_and_preprocess_data()
        
        print("Analyzing depreciation...")
        self.model.analyze_depreciation()
        
        print("Building prediction model...")
        self.model.build_model()
        
        print("Model ready")
    
    def generate_price_prediction_result(self, car_specs):
        """Generate prediction and return as JSON"""
        prediction = self.model.generate_price_prediction(car_specs)
        insights = self.model.get_price_insights(car_specs)
        return {"prediction": prediction, "insights": insights}
    
    def generate_depreciation_result(self, car_specs, years_to_project):
        """Generate depreciation projection as JSON"""
        projection = self.model.create_depreciation_projection(car_specs, years_to_project)
        
        # Convert to list for JSON serialization
        years = list(range(car_specs['tahun'], car_specs['tahun'] + years_to_project + 1))
        values = list(projection.values())
        
        # Calculate percentage changes
        initial_value = values[0]
        percentages = [((value - initial_value) / initial_value * 100) for value in values]
        
        return {
            "years": years,
            "values": values,
            "percentages": percentages
        }
    
    def get_data_insights(self):
        """Get data insights as JSON"""
        if not hasattr(self.model, 'df') or self.model.df is None:
            return {"error": "Data not loaded"}
        
        # Basic stats
        stats = {
            "total_records": len(self.model.df),
            "unique_brands": self.model.df['merek'].nunique(),
            "unique_models": self.model.df['model'].nunique(),
            "avg_price": int(self.model.df['harga'].mean())
        }
        
        # Top brands by price
        top_brands = self.model.df['merek'].value_counts().head(10).index.tolist()
        brand_df = self.model.df[self.model.df['merek'].isin(top_brands)]
        brand_price = brand_df.groupby('merek')['harga'].mean().sort_values(ascending=False).reset_index()
        brand_data = brand_price.to_dict(orient='records')
        
        # Price by year
        year_price = self.model.df.groupby('tahun')['harga'].mean().reset_index()
        year_data = year_price.to_dict(orient='records')
        
        # Depreciation data
        depreciation_data = {}
        if hasattr(self.model, 'median_prices_by_year') and self.model.median_prices_by_year:
            depreciation_data = {
                "years": list(self.model.median_prices_by_year.keys()),
                "prices": list(self.model.median_prices_by_year.values())
            }
            
            if hasattr(self.model, 'depreciation_rates') and self.model.depreciation_rates:
                depreciation_data["avg_rate"] = sum(self.model.depreciation_rates.values()) / len(self.model.depreciation_rates)
        
        return {
            "stats": stats,
            "brand_data": brand_data,
            "year_data": year_data,
            "depreciation_data": depreciation_data
        }

# Initialize the deployment
deployment = FlaskDeployment()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    car_specs = data.get('car_specs', {})
    result = deployment.generate_price_prediction_result(car_specs)
    return jsonify(result)

@app.route('/depreciation', methods=['POST'])
def depreciation():
    data = request.get_json()
    car_specs = data.get('car_specs', {})
    years_to_project = data.get('years_to_project', 10)
    result = deployment.generate_depreciation_result(car_specs, years_to_project)
    return jsonify(result)

@app.route('/insights')
def insights():
    result = deployment.get_data_insights()
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)