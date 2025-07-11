import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import datetime
from car_price_model import CarPriceModel  # Assuming you have a class CarPriceModel defined in car_price_model.py

# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

class StreamlitDeployment:
    def __init__(self):
        """Initialize Streamlit deployment interface"""
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
                st.sidebar.success("Loaded existing model")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")
                self.load_and_build_model()
        else:
            self.load_and_build_model()
    
    def load_and_build_model(self):
        """Load data and build model with progress indicators"""
        with st.sidebar.status("Loading and preparing model..."):
            st.sidebar.info("Loading data...")
            self.model.load_and_preprocess_data()
            
            st.sidebar.info("Analyzing depreciation...")
            self.model.analyze_depreciation()
            
            st.sidebar.info("Building prediction model...")
            self.model.build_model()
            
            st.sidebar.success("Model ready")
    
    def run_app(self):
        """Run the Streamlit application"""
        st.title("🚗 Car Price Prediction")
        st.write("Predict used car prices based on specifications")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Price Prediction", "Depreciation Projection", "Data Insights"])
        
        with tab1:
            self.price_prediction_tab()
        
        with tab2:
            self.depreciation_tab()
        
        with tab3:
            self.data_insights_tab()
    
    def price_prediction_tab(self):
        """Tab for basic price prediction"""
        st.header("Predict Car Price")
        
        # Create 3 columns for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand = st.text_input("Car Brand", "Toyota")
            model = st.text_input("Car Model", "Avanza")
            year = st.number_input("Production Year", 
                                min_value=2000, 
                                max_value=datetime.datetime.now().year,
                                value=2015)
        
        with col2:
            engine = st.number_input("Engine Capacity (L)", 
                                 min_value=0.5, 
                                 max_value=6.0, 
                                 value=1.5,
                                 step=0.1)
            
            variant = st.text_input("Variant (e.g., G, E, TRD)", "")
            
        with col3:
            transmission = st.selectbox("Transmission", 
                                    ["", "Manual", "Otomatis"])
            
            fuel_type = st.selectbox("Fuel Type", 
                                 ["", "Bensin", "Diesel", "Hybrid", "Listrik"])
            
            color = st.selectbox("Color", 
                             ["", "Putih", "Hitam", "Silver", "Merah", "Biru", "Abu", "Kuning", "Hijau", "Coklat"])
        
        # Create car specs dictionary
        car_specs = {
            "merek": brand,
            "model": model,
            "tahun": year,
            "mesin_cc": engine
        }
        
        # Add optional specs if provided
        if variant:
            car_specs["varian"] = variant
        if transmission:
            car_specs["transmisi"] = transmission
        if fuel_type:
            car_specs["bahan_bakar"] = fuel_type
        if color:
            car_specs["warna"] = color
        
        # Predict button
        if st.button("Predict Price"):
            with st.spinner("Generating prediction..."):
                # Get prediction and insights
                prediction = self.model.generate_price_prediction(car_specs)
                insights = self.model.get_price_insights(car_specs)
                
                # Display results
                self.display_prediction_results(prediction, insights, car_specs)
    
    def display_prediction_results(self, prediction, insights, car_specs):
        """Display prediction results with visualizations"""
        # Create two columns for results
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Price Prediction")
            
            # Main prediction
            st.metric(
                label="Predicted Price", 
                value=f"Rp {prediction['predicted_price']:,}"
            )
            
            # Price range
            st.write(f"**Price Range:** Rp {prediction['lower_bound']:,} - Rp {prediction['upper_bound']:,}")
            
            # Create a horizontal bar chart for the price range
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Plot the range
            ax.barh(y=0, width=prediction['upper_bound'] - prediction['lower_bound'], 
                    left=prediction['lower_bound'], height=0.5, color='lightblue', alpha=0.6)
            
            # Plot the predicted value
            ax.scatter(prediction['predicted_price'], 0, color='navy', s=100, zorder=5)
            
            # Plot probable prices
            for price in prediction['probable_prices']:
                ax.scatter(price, 0, color='blue', alpha=0.7, s=50, zorder=4)
            
            # Add labels
            ax.text(prediction['lower_bound'], 0, f"Rp {prediction['lower_bound']:,}", 
                    ha='right', va='center', color='gray')
            ax.text(prediction['upper_bound'], 0, f"Rp {prediction['upper_bound']:,}", 
                    ha='left', va='center', color='gray')
            
            # Format the plot
            ax.set_yticks([])
            ax.set_xlabel('Price (IDR)')
            ax.set_title('Price Prediction Range')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Format x-axis with comma separators
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show probable prices
            st.subheader("Probable Prices")
            
            # Create columns for probable prices
            price_cols = st.columns(len(prediction['probable_prices']))
            for i, (col, price) in enumerate(zip(price_cols, prediction['probable_prices'])):
                with col:
                    st.metric(f"Option {i+1}", f"Rp {price:,}")
        
        with col2:
            st.subheader("Market Comparison")
            
            # Market comparison
            if "market_comparison" in insights and insights["market_comparison"]:
                mc = insights["market_comparison"]
                
                # Create a comparison table
                comparison_data = []
                
                if "similar_models_median" in mc:
                    comparison_data.append({
                        "Type": "Similar Models Median",
                        "Price": mc['similar_models_median'],
                        "Count": mc.get('similar_models_count', 'N/A')
                    })
                
                if "same_year_median" in mc:
                    comparison_data.append({
                        "Type": "Same Year & Model Median",
                        "Price": mc['same_year_median'],
                        "Count": mc.get('same_year_count', 'N/A')
                    })
                
                comparison_data.append({
                    "Type": "Your Car Prediction",
                    "Price": prediction['predicted_price'],
                    "Count": "N/A"
                })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df["Price"] = comparison_df["Price"].apply(lambda x: f"Rp {x:,}")
                    st.table(comparison_df)
                
                # Market assessment
                if "assessment" in mc:
                    assessment_color = "green" if "good deal" in mc["assessment"] else \
                                    "red" if "overpriced" in mc["assessment"] else "blue"
                    
                    st.markdown(f"**Assessment:** <span style='color:{assessment_color}'>{mc['assessment']}</span>", 
                                unsafe_allow_html=True)
                
                # Price difference percentage
                if "diff_from_median_percent" in mc:
                    diff = mc["diff_from_median_percent"]
                    diff_text = f"{diff}% {'higher' if diff > 0 else 'lower'} than market median"
                    diff_color = "red" if diff > 10 else "green" if diff < -5 else "blue"
                    
                    st.markdown(f"**Price Difference:** <span style='color:{diff_color}'>{diff_text}</span>", 
                                unsafe_allow_html=True)
            else:
                st.info("Insufficient market data for comparison")
    
    def depreciation_tab(self):
        """Tab for depreciation projection"""
        st.header("Depreciation Projection")
        st.write("See how your car's value will change over time")
        
        # Create 2 columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.text_input("Car Brand", "Toyota", key="dep_brand")
            model = st.text_input("Car Model", "Avanza", key="dep_model")
            year = st.number_input("Production Year", 
                                min_value=2000, 
                                max_value=datetime.datetime.now().year,
                                value=2015,
                                key="dep_year")
        
        with col2:
            engine = st.number_input("Engine Capacity (L)", 
                                 min_value=0.5, 
                                 max_value=6.0, 
                                 value=1.5,
                                 step=0.1,
                                 key="dep_engine")
            
            years_to_project = st.slider("Years to Project", 
                                     min_value=1, 
                                     max_value=15, 
                                     value=10)
            
            variant = st.text_input("Variant (Optional)", "", key="dep_variant")
        
        # Create car specs dictionary
        car_specs = {
            "merek": brand,
            "model": model,
            "tahun": year,
            "mesin_cc": engine
        }
        
        if variant:
            car_specs["varian"] = variant
        
        # Project button
        if st.button("Project Depreciation"):
            with st.spinner("Generating projection..."):
                # Get depreciation projection
                projection = self.model.create_depreciation_projection(car_specs, years_to_project)
                
                # Display results
                self.display_depreciation_results(projection, car_specs, years_to_project)
    
    def display_depreciation_results(self, projection, car_specs, years_to_project):
        """Display depreciation projection results with visualizations"""
        st.subheader(f"Value Projection for {car_specs['merek']} {car_specs['model']} {car_specs['tahun']}")
        
        # Create DataFrame for visualization
        years = list(range(car_specs['tahun'], car_specs['tahun'] + years_to_project + 1))
        values = list(projection.values())
        
        projection_df = pd.DataFrame({
            'Year': years,
            'Value': values,
            'Age': [i for i in range(years_to_project + 1)]
        })
        
        # Calculate percentage change
        initial_value = projection_df['Value'][0]
        projection_df['Percent'] = ((projection_df['Value'] - initial_value) / initial_value * 100).round(1)
        
        # Display as table
        formatted_df = projection_df.copy()
        formatted_df['Value'] = formatted_df['Value'].apply(lambda x: f"Rp {x:,}")
        formatted_df['Percent'] = formatted_df['Percent'].apply(lambda x: f"{x}%")
        
        st.dataframe(formatted_df, hide_index=True)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the projection
        sns.lineplot(x='Year', y='Value', data=projection_df, marker='o', ax=ax)
        
        # Add labels
        for i, row in projection_df.iterrows():
            ax.text(row['Year'], row['Value'], f"Rp {row['Value']:,}\n({row['Percent']}%)", 
                    ha='center', va='bottom', fontsize=8)
        
        # Format the plot
        ax.set_title(f'Projected Value Depreciation')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value (IDR)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis with comma separators
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Value retention analysis
        retention_5yr = (projection_df[projection_df['Age'] == min(5, years_to_project)]['Value'].values[0] / initial_value) * 100
        
        st.subheader("Value Retention Analysis")
        st.write(f"After 5 years, this car is projected to retain **{retention_5yr:.1f}%** of its original value.")
        
        # Provide context
        if retention_5yr > 60:
            st.success("This car has excellent value retention compared to the market average.")
        elif retention_5yr > 40:
            st.info("This car has average value retention.")
        else:
            st.warning("This car depreciates faster than the market average.")
    
    def data_insights_tab(self):
        """Tab for data insights"""
        st.header("Market Data Insights")
        
        if hasattr(self.model, 'df') and self.model.df is not None:
            # Data overview
            st.subheader("Dataset Overview")
            
            # Key statistics
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Total Records", f"{len(self.model.df):,}")
            
            with stats_col2:
                st.metric("Brands", f"{self.model.df['merek'].nunique()}")
            
            with stats_col3:
                st.metric("Models", f"{self.model.df['model'].nunique()}")
            
            with stats_col4:
                avg_price = int(self.model.df['harga'].mean())
                st.metric("Average Price", f"Rp {avg_price:,}")
            
            # Create tabs for different insights
            insight_tab1, insight_tab2, insight_tab3 = st.tabs(["Price by Brand", "Price by Year", "Depreciation Trend"])
            
            with insight_tab1:
                # Price by brand visualization
                st.subheader("Average Price by Brand")
                
                # Get top brands by count
                top_brands = self.model.df['merek'].value_counts().head(10).index.tolist()
                
                # Filter for top brands
                brand_df = self.model.df[self.model.df['merek'].isin(top_brands)]
                
                # Calculate average price by brand
                brand_price = brand_df.groupby('merek')['harga'].mean().sort_values(ascending=False).reset_index()
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='merek', y='harga', data=brand_price, ax=ax)
                
                # Format the plot
                ax.set_title('Average Price by Brand')
                ax.set_xlabel('Brand')
                ax.set_ylabel('Average Price (IDR)')
                
                # Format y-axis with comma separators
                import matplotlib.ticker as ticker
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with insight_tab2:
                # Price by year visualization
                st.subheader("Average Price by Year")
                
                # Calculate average price by year
                year_price = self.model.df.groupby('tahun')['harga'].mean().reset_index()
                
                # Create line chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(x='tahun', y='harga', data=year_price, marker='o', ax=ax)
                
                # Format the plot
                ax.set_title('Average Price by Year')
                ax.set_xlabel('Year')
                ax.set_ylabel('Average Price (IDR)')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Format y-axis with comma separators
                import matplotlib.ticker as ticker
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with insight_tab3:
                # Depreciation trend
                st.subheader("Depreciation Trend")
                
                if hasattr(self.model, 'median_prices_by_year') and self.model.median_prices_by_year:
                    # Convert to DataFrame
                    depreciation_df = pd.DataFrame({
                        'Year': list(self.model.median_prices_by_year.keys()),
                        'Median Price': list(self.model.median_prices_by_year.values())
                    })
                    
                    # Sort by year
                    depreciation_df = depreciation_df.sort_values('Year')
                    
                    # Create line chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(x='Year', y='Median Price', data=depreciation_df, marker='o', ax=ax)
                    
                    # Format the plot
                    ax.set_title('Median Price by Year (Depreciation Trend)')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Median Price (IDR)')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Format y-axis with comma separators
                    import matplotlib.ticker as ticker
                    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Calculate and display average annual depreciation
                    if hasattr(self.model, 'depreciation_rates') and self.model.depreciation_rates:
                        avg_depreciation = sum(self.model.depreciation_rates.values()) / len(self.model.depreciation_rates)
                        st.info(f"Average annual depreciation rate: {avg_depreciation:.2%}")
                else:
                    st.info("Depreciation data not available")
        else:
            st.info("Please load data first")

def streamlit_app():
    """Main function to run the Streamlit app"""
    deployment = StreamlitDeployment()
    deployment.run_app()

if __name__ == "__main__":
    streamlit_app()