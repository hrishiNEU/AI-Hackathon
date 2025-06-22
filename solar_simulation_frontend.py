import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Solar Panel Grid Maintenance System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-danger { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class SolarPanelGridSimulator:
    def __init__(self, grid_size_km=10, panels_per_km2=1000):
        """Initialize solar panel grid simulator"""
        self.grid_size_km = grid_size_km
        self.panels_per_km2 = panels_per_km2
        self.total_panels = grid_size_km ** 2 * panels_per_km2
        self.grid_coords = self._generate_grid_coordinates()

    def _generate_grid_coordinates(self):
        """Generate coordinates for each panel in the grid"""
        coords = []
        panels_per_side = int(np.sqrt(self.panels_per_km2))

        for i in range(self.grid_size_km):
            for j in range(self.grid_size_km):
                for pi in range(panels_per_side):
                    for pj in range(panels_per_side):
                        x = i + pi / panels_per_side
                        y = j + pj / panels_per_side
                        panel_id = len(coords)
                        coords.append({
                            'panel_id': panel_id,
                            'x_coord': x,
                            'y_coord': y,
                            'sector': f"S_{i}_{j}"
                        })
        return pd.DataFrame(coords)

    def generate_environmental_data(self, days=7):
        """Generate environmental and operational data for the solar panel grid"""
        np.random.seed(42)  # For reproducible results

        data = []
        base_date = datetime.now() - timedelta(days=days)

        for day in range(days):
            current_date = base_date + timedelta(days=day)

            # Daily weather patterns
            season_factor = np.sin(2 * np.pi * day / 365) * 0.3 + 0.7
            daily_temp_base = 25 + season_factor * 15 + np.random.normal(0, 3)
            daily_humidity = max(30, min(90, 60 + np.random.normal(0, 15)))
            wind_speed = max(0, np.random.exponential(8))

            # Rain/dust storm events
            rain_probability = 0.1 + 0.05 * np.sin(2 * np.pi * day / 365)
            dust_storm_probability = 0.02

            has_rain = np.random.random() < rain_probability
            has_dust_storm = np.random.random() < dust_storm_probability

            for panel_idx, panel in self.grid_coords.iterrows():
                panel_id = panel['panel_id']
                x, y = panel['x_coord'], panel['y_coord']

                # Location-based factors
                edge_factor = min(x, y, self.grid_size_km - x, self.grid_size_km - y) / (self.grid_size_km / 2)
                proximity_to_road = 1 - min(1, np.sqrt((x - self.grid_size_km / 2) ** 2 + (y - 0.5) ** 2) / 3)

                # Dust accumulation factors
                base_dust_rate = 0.02 + 0.01 * np.sin(2 * np.pi * day / 7)
                dust_accumulation = base_dust_rate * (day + 1)
                dust_accumulation += np.random.exponential(0.05)

                if has_dust_storm:
                    dust_accumulation += np.random.exponential(0.15)
                dust_accumulation *= (1.5 - edge_factor)
                dust_accumulation += proximity_to_road * np.random.uniform(0.05, 0.15)

                # Rain cleaning effect
                if has_rain:
                    rain_intensity = np.random.exponential(5)
                    cleaning_effect = min(0.9, rain_intensity / 8)
                    panel_accessibility = np.random.uniform(0.5, 1.0)
                    dust_accumulation *= (1 - cleaning_effect * panel_accessibility)

                # Image analysis features
                base_cleanliness = 0.95 - (days - day) * 0.01
                surface_reflectivity = max(0.3, base_cleanliness - dust_accumulation * 0.5)
                dirt_coverage_percent = min(50, dust_accumulation * 50)
                panel_discoloration = np.random.beta(1, 10) * 0.1
                hotspot_detected = np.random.random() < 0.03

                # Electrical measurements
                nominal_voltage = 48.0
                temperature_effect = max(0.85, 1 - (daily_temp_base - 25) * 0.003)
                cleanliness_effect = max(0.75, surface_reflectivity)
                age_degradation = max(0.95, 1 - day * 0.00005)

                current_voltage = nominal_voltage * temperature_effect * cleanliness_effect * age_degradation
                current_voltage += np.random.normal(0, 0.3)

                power_output = current_voltage * 8.33
                power_output *= np.random.uniform(0.98, 1.02)

                # Efficiency calculation
                theoretical_max_power = 400
                efficiency = max(75, min(98, (power_output / theoretical_max_power) * 100))

                # Maintenance needs prediction
                cleaning_threshold_base = 0.15
                efficiency_factor = max(0, (90 - efficiency) / 100)
                dirt_factor = dirt_coverage_percent / 100
                time_factor = min(0.3, day / 7)

                cleaning_probability = cleaning_threshold_base + efficiency_factor + dirt_factor + time_factor
                needs_cleaning = np.random.random() < cleaning_probability

                urgency_score = (
                    max(0, 90 - efficiency) * 0.5 +
                    dirt_coverage_percent * 0.4 +
                    (1 - surface_reflectivity) * 50 * 0.3 +
                    time_factor * 20
                )

                data.append({
                    'date': current_date,
                    'panel_id': panel_id,
                    'x_coord': x,
                    'y_coord': y,
                    'sector': panel['sector'],
                    'temperature': daily_temp_base + np.random.normal(0, 1),
                    'humidity': daily_humidity,
                    'wind_speed': wind_speed,
                    'has_rain': has_rain,
                    'has_dust_storm': has_dust_storm,
                    'surface_reflectivity': surface_reflectivity,
                    'dirt_coverage_percent': dirt_coverage_percent,
                    'panel_discoloration': panel_discoloration,
                    'hotspot_detected': hotspot_detected,
                    'voltage': current_voltage,
                    'current': power_output / current_voltage if current_voltage > 0 else 0,
                    'power_output': power_output,
                    'efficiency': efficiency,
                    'days_since_last_cleaning': (current_date - base_date).days,
                    'proximity_to_road': proximity_to_road,
                    'edge_factor': edge_factor,
                    'needs_cleaning': needs_cleaning,
                    'urgency_score': urgency_score
                })

        return pd.DataFrame(data)

class SolarMaintenancePredictor:
    def __init__(self):
        self.efficiency_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.cleaning_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, data):
        """Prepare features for machine learning models"""
        feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'surface_reflectivity',
            'dirt_coverage_percent', 'panel_discoloration', 'voltage', 'current',
            'days_since_last_cleaning', 'proximity_to_road', 'edge_factor'
        ]

        # Add binary features
        data['has_rain_int'] = data['has_rain'].astype(int)
        data['has_dust_storm_int'] = data['has_dust_storm'].astype(int)
        data['hotspot_detected_int'] = data['hotspot_detected'].astype(int)

        feature_columns.extend(['has_rain_int', 'has_dust_storm_int', 'hotspot_detected_int'])

        return data[feature_columns].fillna(0)

    def train_models(self, data):
        """Train efficiency prediction and cleaning classification models"""
        X = self.prepare_features(data)
        X_scaled = self.scaler.fit_transform(X)

        y_cleaning = data['needs_cleaning']
        class_counts = y_cleaning.value_counts()

        if len(class_counts) < 2:
            n_samples = len(data)
            artificial_cleaning = np.random.choice([True, False], size=n_samples, p=[0.7, 0.3])
            y_cleaning = pd.Series(artificial_cleaning)

        # Train models
        y_efficiency = data['efficiency']
        self.efficiency_model.fit(X_scaled, y_efficiency)
        self.cleaning_model.fit(X_scaled, y_cleaning)

        self.is_trained = True

        # Model evaluation
        X_train, X_test, y_eff_train, y_eff_test = train_test_split(
            X_scaled, y_efficiency, test_size=0.2, random_state=42
        )
        _, _, y_clean_train, y_clean_test = train_test_split(
            X_scaled, y_cleaning, test_size=0.2, random_state=42
        )

        eff_pred = self.efficiency_model.predict(X_test)
        eff_rmse = np.sqrt(mean_squared_error(y_eff_test, eff_pred))

        clean_pred = self.cleaning_model.predict(X_test)

        return {
            'efficiency_rmse': eff_rmse,
            'cleaning_accuracy': np.mean(clean_pred == y_clean_test)
        }

    def predict_maintenance_needs(self, data):
        """Predict efficiency and cleaning needs for new data"""
        if not self.is_trained:
            raise ValueError("Models must be trained first!")

        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)

        predicted_efficiency = self.efficiency_model.predict(X_scaled)
        cleaning_probability = self.cleaning_model.predict_proba(X_scaled)[:, 1]
        needs_cleaning_pred = self.cleaning_model.predict(X_scaled)

        feature_importance = self.cleaning_model.feature_importances_

        results = data.copy()
        results['predicted_efficiency'] = predicted_efficiency
        results['cleaning_probability'] = cleaning_probability
        results['predicted_needs_cleaning'] = needs_cleaning_pred

        return results, feature_importance

@st.cache_data
def load_simulation_data(grid_size_km, panels_per_km2, days):
    """Cache the simulation data to avoid recomputation"""
    grid_simulator = SolarPanelGridSimulator(grid_size_km=grid_size_km, panels_per_km2=panels_per_km2)
    data = grid_simulator.generate_environmental_data(days=days)
    
    predictor = SolarMaintenancePredictor()
    performance = predictor.train_models(data)
    data_with_predictions, feature_importance = predictor.predict_maintenance_needs(data)
    
    return data_with_predictions, feature_importance, performance, grid_simulator

def create_plotly_visualizations(data):
    """Create interactive Plotly visualizations"""
    latest_data = data[data['date'] == data['date'].max()].copy()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Efficiency vs Dirt Coverage',
            'Voltage Distribution',
            'Daily Average Efficiency',
            'Power Output vs Temperature',
            'Surface Reflectivity by Status',
            'Cleaning Probability Distribution',
            'Grid Efficiency Map',
            'Urgency Score Distribution',
            'Panel Performance Over Time'
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # 1. Efficiency vs Dirt Coverage
    fig.add_trace(
        go.Scatter(
            x=data['dirt_coverage_percent'],
            y=data['efficiency'],
            mode='markers',
            marker=dict(
                color=data['cleaning_probability'],
                colorscale='Reds',
                size=4,
                opacity=0.6
            ),
            name='Panels'
        ),
        row=1, col=1
    )
    
    # 2. Voltage Distribution
    fig.add_trace(
        go.Histogram(
            x=data['voltage'],
            nbinsx=30,
            name='Voltage',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # 3. Daily Average Efficiency
    daily_efficiency = data.groupby('date')['efficiency'].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=daily_efficiency['date'],
            y=daily_efficiency['efficiency'],
            mode='lines+markers',
            name='Avg Efficiency',
            line=dict(width=2)
        ),
        row=1, col=3
    )
    
    fig.update_layout(height=1000, showlegend=False)
    return fig

def main():
    """Main Streamlit application"""
    st.title("☀️ Solar Panel Grid Maintenance System")
    st.markdown("*Predictive maintenance dashboard for solar panel grid monitoring*")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    grid_size_km = st.sidebar.slider("Grid Size (km)", min_value=1, max_value=5, value=3, 
                                    help="Side length of the square grid")
    
    panels_per_km2 = st.sidebar.slider("Panels per km²", min_value=100, max_value=1000, value=300, step=50,
                                      help="Number of panels per square kilometer")
    
    days = st.sidebar.slider("Simulation Days", min_value=3, max_value=14, value=7,
                            help="Number of days to simulate")
    
    # Load data
    with st.spinner("🔄 Running simulation..."):
        data, feature_importance, performance, grid_simulator = load_simulation_data(
            grid_size_km, panels_per_km2, days
        )
    
    latest_data = data[data['date'] == data['date'].max()].copy()
    
    # Main dashboard
    st.header("📊 Grid Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_efficiency = latest_data['efficiency'].mean()
        efficiency_status = "🟢" if avg_efficiency > 90 else "🟡" if avg_efficiency > 85 else "🔴"
        st.metric(
            label="Average Efficiency",
            value=f"{avg_efficiency:.1f}%",
            delta=f"{efficiency_status}"
        )
    
    with col2:
        total_power = latest_data['power_output'].sum() / 1000
        st.metric(
            label="Total Power Output",
            value=f"{total_power:.1f} MW",
            delta=f"💡 {grid_simulator.total_panels:,} panels"
        )
    
    with col3:
        panels_needing_cleaning = latest_data['needs_cleaning'].sum()
        cleaning_percentage = (panels_needing_cleaning / len(latest_data)) * 100
        st.metric(
            label="Panels Need Cleaning",
            value=f"{panels_needing_cleaning:,}",
            delta=f"{cleaning_percentage:.1f}%"
        )
    
    with col4:
        high_priority = len(latest_data[
            (latest_data['urgency_score'] > 40) | 
            (latest_data['efficiency'] < 85)
        ])
        st.metric(
            label="High Priority Issues",
            value=f"{high_priority}",
            delta="🚨 Immediate attention" if high_priority > 0 else "✅ All good"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analytics", "🗺️ Grid Map", "🔧 Maintenance", "📋 Reports"])
    
    with tab1:
        st.subheader("Performance Analytics")
        
        # Interactive plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Efficiency vs Dirt Coverage
            fig1 = px.scatter(
                data, 
                x='dirt_coverage_percent', 
                y='efficiency',
                color='cleaning_probability',
                title='Efficiency vs Dirt Coverage',
                labels={'dirt_coverage_percent': 'Dirt Coverage (%)', 'efficiency': 'Efficiency (%)'},
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Daily efficiency trend
            daily_efficiency = data.groupby('date')['efficiency'].mean().reset_index()
            fig2 = px.line(
                daily_efficiency, 
                x='date', 
                y='efficiency',
                title='Daily Average Grid Efficiency',
                labels={'efficiency': 'Efficiency (%)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Additional analytics
        col3, col4 = st.columns(2)
        
        with col3:
            # Power output distribution
            fig3 = px.histogram(
                data, 
                x='power_output',
                title='Power Output Distribution',
                labels={'power_output': 'Power Output (W)'},
                nbins=40
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            # Efficiency by cleaning status
            fig4 = px.box(
                data, 
                x='needs_cleaning', 
                y='efficiency',
                title='Efficiency by Cleaning Status',
                labels={'needs_cleaning': 'Needs Cleaning', 'efficiency': 'Efficiency (%)'}
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab2:
        st.subheader("Grid Spatial Analysis")
        
        # Grid heatmap
        fig_map = px.scatter(
            latest_data,
            x='x_coord',
            y='y_coord',
            color='efficiency',
            size='urgency_score',
            hover_data=['panel_id', 'sector', 'power_output'],
            title='Grid Efficiency Map',
            labels={'x_coord': 'X Coordinate (km)', 'y_coord': 'Y Coordinate (km)'},
            color_continuous_scale='RdYlGn'
        )
        fig_map.update_layout(height=600)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Sector performance
        sector_performance = latest_data.groupby('sector').agg({
            'efficiency': 'mean',
            'needs_cleaning': 'sum',
            'urgency_score': 'mean'
        }).reset_index()
        
        st.subheader("Sector Performance Summary")
        st.dataframe(
            sector_performance.sort_values('efficiency', ascending=False),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Maintenance Dashboard")
        
        # High priority panels
        high_priority_panels = latest_data[
            (latest_data['urgency_score'] > 40) | 
            (latest_data['efficiency'] < 85)
        ].sort_values('urgency_score', ascending=False)
        
        if len(high_priority_panels) > 0:
            st.error(f"🚨 {len(high_priority_panels)} panels require immediate attention!")
            
            st.dataframe(
                high_priority_panels[['panel_id', 'sector', 'efficiency', 'urgency_score', 'needs_cleaning']].head(10),
                use_container_width=True
            )
        else:
            st.success("✅ No panels require immediate attention.")
        
        # Cleaning schedule
        st.subheader("Cleaning Schedule")
        
        cleaning_needed = latest_data[latest_data['needs_cleaning'] == True]
        if len(cleaning_needed) > 0:
            sector_cleaning = cleaning_needed.groupby('sector').agg({
                'panel_id': 'count',
                'urgency_score': 'mean'
            }).sort_values('urgency_score', ascending=False).reset_index()
            
            sector_cleaning.columns = ['Sector', 'Panels to Clean', 'Avg Urgency Score']
            
            st.dataframe(sector_cleaning, use_container_width=True)
        else:
            st.success("✅ No panels currently need cleaning.")
        
        # Feature importance
        st.subheader("Predictive Model Insights")
        
        feature_names = [
            'temperature', 'humidity', 'wind_speed', 'surface_reflectivity',
            'dirt_coverage_percent', 'panel_discoloration', 'voltage', 'current',
            'days_since_last_cleaning', 'proximity_to_road', 'edge_factor',
            'has_rain', 'has_dust_storm', 'hotspot_detected'
        ]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            importance_df.head(8),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top Factors Affecting Cleaning Needs'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab4:
        st.subheader("Comprehensive Report")
        
        # Performance summary
        st.markdown("### 📊 Performance Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Grid Statistics:**
            - Total Panels: {len(latest_data):,}
            - Grid Area: {grid_size_km}×{grid_size_km} km²
            - Average Efficiency: {latest_data['efficiency'].mean():.1f}%
            - Min Efficiency: {latest_data['efficiency'].min():.1f}%
            - Max Efficiency: {latest_data['efficiency'].max():.1f}%
            """)
        
        with col2:
            st.markdown(f"""
            **Maintenance Status:**
            - Panels Needing Cleaning: {latest_data['needs_cleaning'].sum():,}
            - High Priority Issues: {len(high_priority_panels)}
            - Average Urgency Score: {latest_data['urgency_score'].mean():.1f}
            - Hotspots Detected: {latest_data['hotspot_detected'].sum()}
            """)
        
        # Cost analysis
        st.markdown("### 💰 Cost Analysis")
        
        cleaning_cost_per_panel = 5  # USD
        lost_revenue_per_percent_per_day = 2  # USD per panel per efficiency percent lost per day
        
        cleaning_costs = len(cleaning_needed) * cleaning_cost_per_panel
        efficiency_loss = max(0, 95 - latest_data['efficiency'].mean())
        daily_revenue_loss = len(latest_data) * efficiency_loss * lost_revenue_per_percent_per_day
        
        st.markdown(f"""
        - **Immediate cleaning cost:** ${cleaning_costs:,.2f}
        - **Daily revenue loss from efficiency:** ${daily_revenue_loss:,.2f}
        - **ROI of cleaning (1 day):** {(daily_revenue_loss / max(1, cleaning_costs) * 100):.1f}%
        """)
        
        # Download report
        if st.button("📥 Download Full Report"):
            # Create a comprehensive report
            report_data = {
                'Grid Summary': {
                    'Total Panels': len(latest_data),
                    'Grid Area (km²)': grid_size_km ** 2,
                    'Average Efficiency (%)': latest_data['efficiency'].mean(),
                    'Total Power Output (MW)': latest_data['power_output'].sum() / 1000
                },
                'Maintenance Needs': {
                    'Panels Needing Cleaning': len(cleaning_needed),
                    'High Priority Issues': len(high_priority_panels),
                    'Cleaning Cost ($)': cleaning_costs,
                    'Daily Revenue Loss ($)': daily_revenue_loss
                }
            }
            
            st.success("Report generated! (In a real app, this would trigger a download)")
            st.json(report_data)

if __name__ == "__main__":
    main()