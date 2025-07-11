import re
import pandas as pd
import numpy as np
import datetime
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class CarPriceModel:
    def __init__(self, data_path="data/mobil123_data.csv"):
        """Initialize the model with path to scraped data"""
        self.data_path = data_path
        self.df = None
        self.model = None
        self.preprocessor = None
        self.current_year = datetime.datetime.now().year
        self.minimum_year = self.current_year - 20  # For 20 years of depreciation analysis
        self.depreciation_rates = {}
        self.median_prices_by_year = {}
        self.median_prices_by_brand_model_year = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the scraped data"""
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Basic cleaning
        print(f"Original data shape: {self.df.shape}")
        
        # Process price from harga_text instead of harga
        if 'harga_text' in self.df.columns:
            self.df['harga_clean'] = self.df['harga_text'].apply(self.clean_price_indonesian)
            # Replace harga with properly cleaned values
            self.df['harga'] = self.df['harga_clean']
        
        # Drop rows with missing essential data
        self.df = self.df.dropna(subset=['harga', 'tahun'])
        
        # Filter out rows with invalid prices or years
        self.df = self.df[self.df['harga'] > 10000]  # Remove unrealistically low prices
        self.df = self.df[self.df['harga'] < 10000000000]  # Cap at 10 billion IDR
        self.df = self.df[self.df['tahun'] >= 1990]  # Remove very old or invalid years
        self.df = self.df[self.df['tahun'] <= self.current_year]  # No future years
        
        # Handle missing values in categorical columns
        for col in ['merek', 'model', 'transmisi', 'bahan_bakar', 'warna']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
        
        # Convert engine capacity to numeric if it's not
        if 'mesin_cc' in self.df.columns:
            self.df['mesin_cc'] = pd.to_numeric(self.df['mesin_cc'], errors='coerce')
            self.df['mesin_cc'] = self.df['mesin_cc'].fillna(self.df['mesin_cc'].median())
        
        # Create age feature (years since production)
        self.df['age'] = self.current_year - self.df['tahun']
        
        # Ensure brand and model are properly formatted
        if 'merek' in self.df.columns and 'model' in self.df.columns:
            self.df['merek'] = self.df['merek'].str.lower().str.strip().str.capitalize()
            self.df['model'] = self.df['model'].str.lower().str.strip().str.capitalize()
            
            # Create brand_model feature for better grouping
            self.df['brand_model'] = self.df['merek'] + " " + self.df['model']
        
        print(f"Cleaned data shape: {self.df.shape}")
        return self.df
    
    def clean_price_indonesian(self, price_text):
        """
        Clean Indonesian price format (e.g., 'Rp 1.040.000.000') to integer
        In Indonesian format, periods (.) are thousand separators and commas (,) are decimal separators
        """
        if not price_text or not isinstance(price_text, str):
            return 0
            
        # Remove 'Rp' and any non-numeric characters except periods and commas
        price_clean = re.sub(r'[^0-9.,]', '', price_text)
        
        # Remove periods (thousand separators in Indonesian format)
        price_clean = price_clean.replace('.', '')
        
        # Convert any comma (decimal separator) to period and ensure it's a valid number
        if ',' in price_clean:
            parts = price_clean.split(',')
            if len(parts) > 1:
                # Keep only the whole number part for car prices
                price_clean = parts[0]
        
        # Convert to integer
        try:
            return int(price_clean)
        except ValueError:
            return 0
    
    def analyze_depreciation(self):
        """Analyze price depreciation over time"""
        if self.df is None:
            self.load_and_preprocess_data()
            
        print("\n=== Price Depreciation Analysis ===")
        
        # Check data availability for depreciation analysis
        year_counts = self.df['tahun'].value_counts().sort_index()
        available_years = year_counts.index.tolist()
        print(f"Available years in data: {min(available_years)} to {max(available_years)}")
        
        # Calculate median prices by year
        year_prices = self.df.groupby('tahun')['harga'].median().reset_index()
        self.median_prices_by_year = dict(zip(year_prices['tahun'], year_prices['harga']))
        
        # Calculate depreciation rates between consecutive years
        for i in range(len(year_prices) - 1):
            current_year = year_prices.iloc[i]['tahun']
            next_year = year_prices.iloc[i+1]['tahun']
            
            if next_year == current_year + 1:  # Only for consecutive years
                current_price = year_prices.iloc[i]['harga']
                next_price = year_prices.iloc[i+1]['harga']
                
                if current_price > 0 and next_price > 0:
                    # For depreciation, newer cars should be more expensive
                    # So depreciation rate should be calculated from newer to older
                    if next_year > current_year:
                        depreciation_rate = (next_price - current_price) / next_price
                    else:
                        depreciation_rate = (current_price - next_price) / current_price
                    
                    # Ensure depreciation rate is positive (represents loss of value)
                    self.depreciation_rates[current_year] = abs(depreciation_rate)
        
        # Calculate average annual depreciation rate
        if self.depreciation_rates:
            avg_depreciation = sum(self.depreciation_rates.values()) / len(self.depreciation_rates)
            print(f"Average annual depreciation rate: {avg_depreciation:.2%}")
        else:
            # Default depreciation rate if we can't calculate from data
            avg_depreciation = 0.1  # 10% annual depreciation as a default
            print(f"Using default annual depreciation rate: {avg_depreciation:.2%}")
        
        # Calculate median prices by brand, model and year for more precise depreciation
        if 'brand_model' in self.df.columns:
            groups = self.df.groupby(['brand_model', 'tahun'])['harga'].median().reset_index()
            
            for brand_model in groups['brand_model'].unique():
                brand_model_data = groups[groups['brand_model'] == brand_model]
                
                # Only consider brand_models with sufficient data points
                if len(brand_model_data) >= 3:
                    # Create a mapping of year to median price for this brand_model
                    year_to_price = dict(zip(brand_model_data['tahun'], brand_model_data['harga']))
                    self.median_prices_by_brand_model_year[brand_model] = year_to_price
        
        # Plot depreciation trend
        self._plot_depreciation_trend(year_prices)
        
        return self.depreciation_rates, self.median_prices_by_year
    
    def _plot_depreciation_trend(self, year_prices):
        """Plot the depreciation trend over years"""
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='tahun', y='harga', data=year_prices, marker='o')
        plt.title('Median Car Prices by Year (Depreciation Trend)')
        plt.xlabel('Year')
        plt.ylabel('Median Price (IDR)')
        plt.grid(True)
        plt.savefig('depreciation_trend.png')
        plt.close()
        
        print("Depreciation trend plot saved as 'depreciation_trend.png'")
    
    def build_model(self):
        """Build a generative AI model for price prediction"""
        if self.df is None:
            self.load_and_preprocess_data()
            
        if not self.depreciation_rates:
            self.analyze_depreciation()
            
        print("\n=== Building Generative Price Prediction Model ===")
        
        # Select features and target
        features = ['tahun', 'merek', 'model', 'varian', 'mesin_cc', 'transmisi', 'bahan_bakar', 'warna']
        target = 'harga'
        
        # Keep only available features
        features = [f for f in features if f in self.df.columns]
        
        # Split data
        X = self.df[features]
        y = self.df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessor
        numeric_features = [f for f in features if X[f].dtype in ['int64', 'float64']]
        categorical_features = [f for f in features if X[f].dtype == 'object']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Define model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: Rp {mae:,.0f}")
        print(f"Root Mean Squared Error: Rp {rmse:,.0f}")
        print(f"R² Score: {r2:.4f}")
        
        self.model = model
        self.preprocessor = preprocessor
        
        # Save model
        joblib.dump(model, 'models/car_price_model.pkl')
        print("Model saved as 'car_price_model.pkl'")
        
        return model
    
    def generate_price_prediction(self, car_specs: Dict) -> Dict:
        """
        Generate price prediction with uncertainty estimates
        
        Args:
            car_specs: Dictionary with car specifications
                       Required keys: merek, model, tahun
                       Optional keys: varian, mesin_cc, transmisi, bahan_bakar, warna
        
        Returns:
            Dictionary with predicted price and confidence interval
        """
        if self.model is None:
            try:
                self.model = joblib.load('car_price_model.pkl')
            except:
                self.build_model()
        
        # Ensure required fields are present
        required_fields = ['merek', 'model', 'tahun']
        for field in required_fields:
            if field not in car_specs:
                return {"error": f"Missing required field: {field}"}
        
        # Create brand_model for lookup
        brand_model = f"{car_specs['merek'].capitalize()} {car_specs['model'].capitalize()}"
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([car_specs])
        
        # Fill missing values
        for col in self.model.feature_names_in_:
            if col not in input_df.columns:
                if col in ['mesin_cc']:
                    input_df[col] = self.df[col].median()
                else:
                    input_df[col] = 'Unknown'
        
        # Get base prediction from ML model
        try:
            base_prediction = self.model.predict(input_df[self.model.feature_names_in_])[0]
        except:
            # Fallback to simpler approach if model prediction fails
            base_prediction = self._fallback_prediction(car_specs, brand_model)
        
        # Add uncertainty based on data availability
        uncertainty_factor = self._calculate_uncertainty(car_specs, brand_model)
        
        # Generate distribution parameters
        std_dev = base_prediction * uncertainty_factor
        
        # Create price range (95% confidence interval)
        lower_bound = max(0, base_prediction - 1.96 * std_dev)
        upper_bound = base_prediction + 1.96 * std_dev
        
        # Generate 5 probable prices within the range
        probable_prices = sorted(np.random.normal(base_prediction, std_dev, 5).tolist())
        probable_prices = [max(0, p) for p in probable_prices]
        
        return {
            "predicted_price": int(base_prediction),
            "lower_bound": int(lower_bound),
            "upper_bound": int(upper_bound),
            "probable_prices": [int(p) for p in probable_prices],
            "uncertainty_factor": round(uncertainty_factor, 2)
        }
    
    def _fallback_prediction(self, car_specs: Dict, brand_model: str) -> float:
        """Fallback prediction method using depreciation rates"""
        year = car_specs['tahun']
        
        # Try to find this exact brand_model and year in our data
        if brand_model in self.median_prices_by_brand_model_year and year in self.median_prices_by_brand_model_year[brand_model]:
            return self.median_prices_by_brand_model_year[brand_model][year]
        
        # Try to find the median price for this year across all cars
        if year in self.median_prices_by_year:
            return self.median_prices_by_year[year]
        
        # If we have no exact match, use depreciation to estimate from closest available year
        if brand_model in self.median_prices_by_brand_model_year:
            # Find closest available year
            available_years = list(self.median_prices_by_brand_model_year[brand_model].keys())
            if available_years:
                closest_year = min(available_years, key=lambda x: abs(x - year))
                known_price = self.median_prices_by_brand_model_year[brand_model][closest_year]
                
                # Apply depreciation
                years_diff = abs(year - closest_year)
                avg_annual_depreciation = 0.1  # Default 10% if we don't have calculated rates
                if self.depreciation_rates:
                    avg_annual_depreciation = sum(self.depreciation_rates.values()) / len(self.depreciation_rates)
                
                if year < closest_year:  # Older car, more depreciation
                    return known_price * ((1 - avg_annual_depreciation) ** years_diff)
                else:  # Newer car, less depreciation
                    return known_price / ((1 - avg_annual_depreciation) ** years_diff)
        
        # Absolute fallback: average price in the dataset
        return self.df['harga'].median()
    
    def _calculate_uncertainty(self, car_specs: Dict, brand_model: str) -> float:
        """Calculate uncertainty factor based on data availability"""
        year = car_specs['tahun']
        base_uncertainty = 0.15  # Start with 15% uncertainty
        
        # Lower uncertainty if we have exact data for this brand_model and year
        if brand_model in self.median_prices_by_brand_model_year and year in self.median_prices_by_brand_model_year[brand_model]:
            base_uncertainty *= 0.7
        
        # Increase uncertainty for very old or very new cars
        age = self.current_year - year
        if age > 15 or age < 1:
            base_uncertainty *= 1.5
        
        # Increase uncertainty if we have few data points for this brand_model
        if brand_model in self.median_prices_by_brand_model_year:
            data_points = len(self.median_prices_by_brand_model_year[brand_model])
            if data_points < 3:
                base_uncertainty *= 1.3
        else:
            # Even more uncertainty if we have no data for this brand_model
            base_uncertainty *= 1.6
        
        return min(0.5, base_uncertainty)  # Cap at 50% to avoid extreme values
    
    def create_depreciation_projection(self, initial_specs: Dict, years_forward: int = 10) -> Dict:
        """
        Project car value depreciation over specified years
        
        Args:
            initial_specs: Initial car specifications
            years_forward: Number of years to project forward
            
        Returns:
            Dictionary with projected values for each year
        """
        if not self.depreciation_rates:
            self.analyze_depreciation()
        
        # Get initial prediction
        initial_price = self.generate_price_prediction(initial_specs)["predicted_price"]
        
        # Use average depreciation rate
        avg_annual_depreciation = 0.1  # Default 10%
        if self.depreciation_rates:
            avg_annual_depreciation = sum(self.depreciation_rates.values()) / len(self.depreciation_rates)
        
        # Generate projections
        projections = {0: initial_price}
        current_price = initial_price
        
        for year in range(1, years_forward + 1):
            # Apply depreciation (value decreases each year)
            current_price = current_price * (1 - avg_annual_depreciation)
            projections[year] = int(current_price)
        
        return projections
    
    def get_price_insights(self, car_specs: Dict) -> Dict:
        """Get insights about the car's price compared to market"""
        prediction = self.generate_price_prediction(car_specs)
        
        # Get median price for same brand and model
        brand_model = f"{car_specs['merek'].capitalize()} {car_specs['model'].capitalize()}"
        
        similar_cars = self.df[(self.df['merek'].str.lower() == car_specs['merek'].lower()) & 
                              (self.df['model'].str.lower() == car_specs['model'].lower())]
        
        same_year_cars = similar_cars[similar_cars['tahun'] == car_specs['tahun']]
        
        insights = {
            "prediction": prediction,
            "market_comparison": {}
        }
        
        if not similar_cars.empty:
            insights["market_comparison"]["similar_models_count"] = len(similar_cars)
            insights["market_comparison"]["similar_models_median"] = int(similar_cars['harga'].median())
            
            if not same_year_cars.empty:
                insights["market_comparison"]["same_year_count"] = len(same_year_cars)
                insights["market_comparison"]["same_year_median"] = int(same_year_cars['harga'].median())
            
            # Calculate if prediction is above or below market median
            market_median = insights["market_comparison"].get("same_year_median", 
                                                            insights["market_comparison"].get("similar_models_median"))
            
            if market_median:
                diff_percent = (prediction["predicted_price"] - market_median) / market_median * 100
                insights["market_comparison"]["diff_from_median_percent"] = round(diff_percent, 1)
                
                if diff_percent < -10:
                    insights["market_comparison"]["assessment"] = "Below market average (potential good deal)"
                elif diff_percent > 10:
                    insights["market_comparison"]["assessment"] = "Above market average (potentially overpriced)"
                else:
                    insights["market_comparison"]["assessment"] = "Close to market average (fair price)"
        
        # Add depreciation projection
        insights["depreciation_projection"] = self.create_depreciation_projection(car_specs, 5)
        
        return insights

class SimpleDeployment:
    def __init__(self, model: CarPriceModel = None):
        """Initialize simple deployment interface"""
        if model is None:
            self.model = CarPriceModel()
            # Try to load existing model, or build if not available
            try:
                self.model.model = joblib.load('car_price_model.pkl')
                self.model.load_and_preprocess_data()
                self.model.analyze_depreciation()
            except:
                self.model.load_and_preprocess_data()
                self.model.analyze_depreciation()
                self.model.build_model()
        else:
            self.model = model
    
    def predict_price(self, car_specs: Dict) -> Dict:
        """Make price prediction for given car specs"""
        return self.model.generate_price_prediction(car_specs)
    
    def get_insights(self, car_specs: Dict) -> Dict:
        """Get comprehensive price insights"""
        return self.model.get_price_insights(car_specs)
    
    def run_interactive_cli(self):
        """Run interactive command-line interface"""
        print("\n=== Car Price Prediction System ===")
        
        while True:
            print("\nOptions:")
            print("1. Predict car price")
            print("2. Get comprehensive price insights")
            print("3. Project depreciation")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == "1" or choice == "2" or choice == "3":
                # Collect car specifications
                brand = input("Enter car brand (e.g., Toyota): ").strip()
                model = input("Enter car model (e.g., Avanza): ").strip()
                year = input("Enter production year (e.g., 2015): ").strip()
                
                try:
                    year = int(year)
                    car_specs = {
                        "merek": brand,
                        "model": model,
                        "tahun": year
                    }
                    
                    # Optional specs
                    engine = input("Enter engine capacity in liters (optional, e.g., 1.5): ").strip()
                    if engine:
                        car_specs["mesin_cc"] = float(engine)
                        
                    variant = input("Enter variant (optional, e.g., G, E, TRD): ").strip()
                    if variant:
                        car_specs["varian"] = variant
                        
                    transmission = input("Enter transmission (optional, Manual/Otomatis): ").strip()
                    if transmission:
                        car_specs["transmisi"] = transmission
                        
                    fuel = input("Enter fuel type (optional, Bensin/Diesel/Hybrid): ").strip()
                    if fuel:
                        car_specs["bahan_bakar"] = fuel
                    
                    if choice == "1":
                        # Simple prediction
                        result = self.predict_price(car_specs)
                        
                        print("\n=== Price Prediction ===")
                        print(f"Car: {brand} {model} {year}")
                        print(f"Predicted Price: Rp {result['predicted_price']:,}")
                        print(f"Price Range: Rp {result['lower_bound']:,} - Rp {result['upper_bound']:,}")
                        print(f"Probable Prices:")
                        for i, price in enumerate(result['probable_prices']):
                            print(f"  Option {i+1}: Rp {price:,}")
                    
                    elif choice == "2":
                        # Comprehensive insights
                        insights = self.get_insights(car_specs)
                        
                        print("\n=== Price Insights ===")
                        print(f"Car: {brand} {model} {year}")
                        print(f"Predicted Price: Rp {insights['prediction']['predicted_price']:,}")
                        print(f"Price Range: Rp {insights['prediction']['lower_bound']:,} - Rp {insights['prediction']['upper_bound']:,}")
                        
                        if "market_comparison" in insights and insights["market_comparison"]:
                            mc = insights["market_comparison"]
                            print("\nMarket Comparison:")
                            
                            if "similar_models_count" in mc:
                                print(f"Similar models in database: {mc['similar_models_count']}")
                            
                            if "similar_models_median" in mc:
                                print(f"Median price for similar models: Rp {mc['similar_models_median']:,}")
                            
                            if "same_year_count" in mc:
                                print(f"Same year & model in database: {mc['same_year_count']}")
                            
                            if "same_year_median" in mc:
                                print(f"Median price for same year & model: Rp {mc['same_year_median']:,}")
                            
                            if "diff_from_median_percent" in mc:
                                print(f"Difference from market median: {mc['diff_from_median_percent']}%")
                            
                            if "assessment" in mc:
                                print(f"Assessment: {mc['assessment']}")
                    
                    elif choice == "3":
                        # Depreciation projection
                        years = input("Enter number of years to project (default 5): ").strip()
                        years = int(years) if years.isdigit() else 5
                        
                        projections = self.model.create_depreciation_projection(car_specs, years)
                        
                        print("\n=== Depreciation Projection ===")
                        print(f"Car: {brand} {model} {year}")
                        print(f"Initial Value: Rp {projections[0]:,}")
                        
                        for i in range(1, years + 1):
                            current_year = year + i
                            percent_change = ((projections[i] - projections[0]) / projections[0]) * 100
                            print(f"Year {current_year} (after {i} years): Rp {projections[i]:,} " + 
                                  f"({percent_change:.1f}% change)")
                
                except ValueError as e:
                    print(f"Error: Invalid input - {e}")
                except Exception as e:
                    print(f"Error: {e}")
            
            elif choice == "4":
                print("Exiting. Thank you for using the Car Price Prediction System!")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main function to run the data processing and modeling"""
    # Load and process data
    car_model = CarPriceModel()
    car_model.load_and_preprocess_data()
    
    # Analyze depreciation
    car_model.analyze_depreciation()
    
    # Build predictive model
    car_model.build_model()
    
    # Deploy simple interface
    deployment = SimpleDeployment(car_model)
    deployment.run_interactive_cli()

if __name__ == "__main__":
    main()