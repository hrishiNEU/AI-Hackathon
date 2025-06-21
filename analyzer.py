import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class SolarPanelGridSimulator:
    def __init__(self, grid_size_km=10, panels_per_km2=1000):
        """
        Initialize solar panel grid simulator

        Args:
            grid_size_km: Side length of square grid in km (default 10km x 10km = 100 sq km)
            panels_per_km2: Number of panels per square kilometer
        """
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

    def generate_environmental_data(self, days=30):
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

                # Drone image analysis simulation (0-1 scale)
                base_cleanliness = 0.95 - (days - day) * 0.01  # Gradual degradation

                # Dust accumulation factors with more realistic variation
                base_dust_rate = 0.01 + 0.005 * np.sin(2 * np.pi * day / 30)  # Monthly cycle
                dust_accumulation = base_dust_rate * (day + 1)  # Cumulative dust
                dust_accumulation += np.random.exponential(0.05)  # Random dust events

                if has_dust_storm:
                    dust_accumulation += np.random.exponential(0.15)
                dust_accumulation *= (1.5 - edge_factor)  # More dust at edges
                dust_accumulation += proximity_to_road * np.random.uniform(0.05, 0.15)  # Road dust variation

                # Rain cleaning effect with variation
                if has_rain:
                    rain_intensity = np.random.exponential(5)
                    cleaning_effect = min(0.9, rain_intensity / 8)
                    # Some panels get cleaned better than others
                    panel_accessibility = np.random.uniform(0.5, 1.0)
                    dust_accumulation *= (1 - cleaning_effect * panel_accessibility)

                # Image analysis features
                surface_reflectivity = max(0.3, base_cleanliness - dust_accumulation * 0.5)
                dirt_coverage_percent = min(50, dust_accumulation * 50)  # Reduced max dirt coverage
                panel_discoloration = np.random.beta(1, 10) * 0.1  # Reduced discoloration
                hotspot_detected = np.random.random() < 0.03  # 3% chance of hotspots

                # Electrical measurements with more realistic efficiency
                nominal_voltage = 48.0  # Volts
                temperature_effect = max(0.85, 1 - (daily_temp_base - 25) * 0.003)  # Reduced temperature impact
                cleanliness_effect = max(0.75, surface_reflectivity)  # Minimum 75% efficiency
                age_degradation = max(0.95, 1 - day * 0.00005)  # Slower degradation

                current_voltage = nominal_voltage * temperature_effect * cleanliness_effect * age_degradation
                current_voltage += np.random.normal(0, 0.3)  # Less measurement noise

                power_output = current_voltage * 8.33  # Assuming 8.33A current for 400W panels
                power_output *= np.random.uniform(0.98, 1.02)  # Smaller variations

                # Efficiency calculation - keep it realistic (75-98%)
                theoretical_max_power = 400  # Watts
                efficiency = max(75, min(98, (power_output / theoretical_max_power) * 100))

                # Maintenance needs prediction with more realistic thresholds
                cleaning_threshold_base = 0.15  # 15% base probability
                efficiency_factor = max(0, (90 - efficiency) / 100)  # Higher probability for lower efficiency
                dirt_factor = dirt_coverage_percent / 100
                time_factor = min(0.3, day / 30)  # Time-based factor

                cleaning_probability = cleaning_threshold_base + efficiency_factor + dirt_factor + time_factor
                needs_cleaning = np.random.random() < cleaning_probability

                urgency_score = (
                        max(0, 90 - efficiency) * 0.5 +  # Reduced weight for efficiency
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

                    # Environmental
                    'temperature': daily_temp_base + np.random.normal(0, 1),
                    'humidity': daily_humidity,
                    'wind_speed': wind_speed,
                    'has_rain': has_rain,
                    'has_dust_storm': has_dust_storm,

                    # Drone image analysis
                    'surface_reflectivity': surface_reflectivity,
                    'dirt_coverage_percent': dirt_coverage_percent,
                    'panel_discoloration': panel_discoloration,
                    'hotspot_detected': hotspot_detected,

                    # Electrical measurements
                    'voltage': current_voltage,
                    'current': power_output / current_voltage if current_voltage > 0 else 0,
                    'power_output': power_output,
                    'efficiency': efficiency,

                    # Maintenance indicators
                    'days_since_last_cleaning': (current_date - base_date).days,
                    'proximity_to_road': proximity_to_road,
                    'edge_factor': edge_factor,
                    'needs_cleaning': needs_cleaning,
                    'urgency_score': urgency_score
                })

        return pd.DataFrame(data)


class SolarMaintenancePredictor:
    def __init__(self):
        self.efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cleaning_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
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

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Check class distribution before training
        y_cleaning = data['needs_cleaning']
        class_counts = y_cleaning.value_counts()
        print(
            f"Class distribution - Needs cleaning: {class_counts.get(True, 0)}, Doesn't need: {class_counts.get(False, 0)}")

        if len(class_counts) < 2:
            # If only one class, create some artificial variation
            print("⚠️  Only one class detected. Creating balanced dataset...")
            n_samples = len(data)
            # Randomly assign some panels as not needing cleaning
            artificial_cleaning = np.random.choice([True, False], size=n_samples, p=[0.7, 0.3])
            y_cleaning = pd.Series(artificial_cleaning)
            print(
                f"Adjusted class distribution - Needs cleaning: {sum(artificial_cleaning)}, Doesn't need: {sum(~artificial_cleaning)}")

        # Train efficiency prediction model
        y_efficiency = data['efficiency']
        self.efficiency_model.fit(X_scaled, y_efficiency)

        # Train cleaning prediction model
        self.cleaning_model.fit(X_scaled, y_cleaning)

        self.is_trained = True

        # Model evaluation
        X_train, X_test, y_eff_train, y_eff_test = train_test_split(
            X_scaled, y_efficiency, test_size=0.2, random_state=42
        )

        _, _, y_clean_train, y_clean_test = train_test_split(
            X_scaled, y_cleaning, test_size=0.2, random_state=42
        )

        # Efficiency model performance
        eff_pred = self.efficiency_model.predict(X_test)
        eff_rmse = np.sqrt(mean_squared_error(y_eff_test, eff_pred))

        # Cleaning model performance
        clean_pred = self.cleaning_model.predict(X_test)

        print("Model Performance:")
        print(f"Efficiency Prediction RMSE: {eff_rmse:.2f}%")
        print("\nCleaning Prediction Classification Report:")
        print(classification_report(y_clean_test, clean_pred))

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

        # Predictions
        predicted_efficiency = self.efficiency_model.predict(X_scaled)
        cleaning_probability = self.cleaning_model.predict_proba(X_scaled)[:, 1]
        needs_cleaning_pred = self.cleaning_model.predict(X_scaled)

        # Feature importance for cleaning decisions
        feature_importance = self.cleaning_model.feature_importances_

        results = data.copy()
        results['predicted_efficiency'] = predicted_efficiency
        results['cleaning_probability'] = cleaning_probability
        results['predicted_needs_cleaning'] = needs_cleaning_pred

        return results, feature_importance


def visualize_results(data, grid_simulator):
    """Create comprehensive visualizations of the solar panel grid analysis"""

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))

    # Get latest data for spatial plots
    latest_data = data[data['date'] == data['date'].max()].copy()

    # 1. Grid efficiency heatmap
    ax1 = plt.subplot(3, 4, 1)
    pivot_eff = latest_data.pivot_table(
        values='efficiency',
        index='y_coord',
        columns='x_coord',
        aggfunc='mean'
    )
    sns.heatmap(pivot_eff, cmap='RdYlGn', center=85, ax=ax1, cbar_kws={'label': 'Efficiency (%)'})
    ax1.set_title('Current Grid Efficiency Distribution')
    ax1.set_xlabel('X Coordinate (km)')
    ax1.set_ylabel('Y Coordinate (km)')

    # 2. Cleaning needs heatmap
    ax2 = plt.subplot(3, 4, 2)
    pivot_clean = latest_data.pivot_table(
        values='cleaning_probability',
        index='y_coord',
        columns='x_coord',
        aggfunc='mean'
    )
    sns.heatmap(pivot_clean, cmap='Reds', ax=ax2, cbar_kws={'label': 'Cleaning Probability'})
    ax2.set_title('Predicted Cleaning Needs')
    ax2.set_xlabel('X Coordinate (km)')
    ax2.set_ylabel('Y Coordinate (km)')

    # 3. Efficiency vs dirt coverage
    ax3 = plt.subplot(3, 4, 3)
    scatter = ax3.scatter(data['dirt_coverage_percent'], data['efficiency'],
                          c=data['cleaning_probability'], cmap='Reds', alpha=0.6)
    ax3.set_xlabel('Dirt Coverage (%)')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Efficiency vs Dirt Coverage')
    plt.colorbar(scatter, ax=ax3, label='Cleaning Probability')

    # 4. Voltage distribution
    ax4 = plt.subplot(3, 4, 4)
    ax4.hist(data['voltage'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(data['voltage'].mean(), color='red', linestyle='--',
                label=f'Mean: {data["voltage"].mean():.2f}V')
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Voltage Distribution Across Grid')
    ax4.legend()

    # 5. Time series of average efficiency
    ax5 = plt.subplot(3, 4, 5)
    daily_efficiency = data.groupby('date')['efficiency'].mean()
    ax5.plot(daily_efficiency.index, daily_efficiency.values, marker='o', linewidth=2)
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Average Efficiency (%)')
    ax5.set_title('Daily Average Grid Efficiency')
    ax5.tick_params(axis='x', rotation=45)

    # 6. Cleaning urgency by sector
    ax6 = plt.subplot(3, 4, 6)
    sector_urgency = latest_data.groupby('sector')['urgency_score'].mean().sort_values(ascending=False)
    top_sectors = sector_urgency.head(10)
    bars = ax6.bar(range(len(top_sectors)), top_sectors.values, color='orange', alpha=0.7)
    ax6.set_xlabel('Sector')
    ax6.set_ylabel('Average Urgency Score')
    ax6.set_title('Top 10 Sectors Needing Attention')
    ax6.set_xticks(range(len(top_sectors)))
    ax6.set_xticklabels(top_sectors.index, rotation=45)

    # 7. Power output vs temperature
    ax7 = plt.subplot(3, 4, 7)
    ax7.scatter(data['temperature'], data['power_output'], alpha=0.5, color='green')
    ax7.set_xlabel('Temperature (°C)')
    ax7.set_ylabel('Power Output (W)')
    ax7.set_title('Power Output vs Temperature')

    # 8. Surface reflectivity distribution
    ax8 = plt.subplot(3, 4, 8)
    clean_panels = data[data['needs_cleaning'] == False]['surface_reflectivity']
    dirty_panels = data[data['needs_cleaning'] == True]['surface_reflectivity']

    ax8.hist(clean_panels, bins=30, alpha=0.7, label='Clean Panels', color='green')
    ax8.hist(dirty_panels, bins=30, alpha=0.7, label='Needs Cleaning', color='red')
    ax8.set_xlabel('Surface Reflectivity')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Surface Reflectivity Distribution')
    ax8.legend()

    # 9. Correlation matrix of key variables
    ax9 = plt.subplot(3, 4, 9)
    corr_vars = ['efficiency', 'dirt_coverage_percent', 'surface_reflectivity',
                 'voltage', 'temperature', 'urgency_score']
    corr_matrix = data[corr_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9)
    ax9.set_title('Feature Correlation Matrix')

    # 10. Efficiency prediction accuracy
    ax10 = plt.subplot(3, 4, 10)
    if 'predicted_efficiency' in data.columns:
        ax10.scatter(data['efficiency'], data['predicted_efficiency'], alpha=0.5)
        min_eff, max_eff = data['efficiency'].min(), data['efficiency'].max()
        ax10.plot([min_eff, max_eff], [min_eff, max_eff], 'r--', label='Perfect Prediction')
        ax10.set_xlabel('Actual Efficiency (%)')
        ax10.set_ylabel('Predicted Efficiency (%)')
        ax10.set_title('Efficiency Prediction Accuracy')
        ax10.legend()

    # 11. Daily cleaning recommendations
    ax11 = plt.subplot(3, 4, 11)
    daily_cleaning = data.groupby('date')['needs_cleaning'].sum()
    ax11.plot(daily_cleaning.index, daily_cleaning.values, marker='s', color='red', linewidth=2)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('Panels Needing Cleaning')
    ax11.set_title('Daily Cleaning Recommendations')
    ax11.tick_params(axis='x', rotation=45)

    # 12. Summary statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    stats_text = f"""
    GRID SUMMARY STATISTICS

    Total Panels: {len(latest_data):,}
    Grid Area: {grid_simulator.grid_size_km}×{grid_simulator.grid_size_km} km²

    Current Performance:
    • Avg Efficiency: {latest_data['efficiency'].mean():.1f}%
    • Min Efficiency: {latest_data['efficiency'].min():.1f}%
    • Max Efficiency: {latest_data['efficiency'].max():.1f}%

    Maintenance Needs:
    • Panels Needing Cleaning: {latest_data['needs_cleaning'].sum():,}
    • Percentage Needing Cleaning: {(latest_data['needs_cleaning'].mean() * 100):.1f}%
    • Avg Urgency Score: {latest_data['urgency_score'].mean():.1f}

    Power Generation:
    • Total Power Output: {latest_data['power_output'].sum() / 1000:.1f} MW
    • Avg Power per Panel: {latest_data['power_output'].mean():.1f} W

    Environmental:
    • Avg Temperature: {latest_data['temperature'].mean():.1f}°C
    • Avg Dirt Coverage: {latest_data['dirt_coverage_percent'].mean():.1f}%
    """

    ax12.text(0.1, 0.9, stats_text, transform=ax12.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()


def generate_maintenance_report(data, feature_importance, feature_names):
    """Generate a comprehensive maintenance report"""

    latest_data = data[data['date'] == data['date'].max()].copy()

    print("=" * 80)
    print("SOLAR PANEL GRID MAINTENANCE REPORT")
    print("=" * 80)
    print(f"Report Date: {latest_data['date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"Grid Coverage: {int(np.sqrt(len(latest_data)))}×{int(np.sqrt(len(latest_data)))} km²")
    print(f"Total Panels Monitored: {len(latest_data):,}")

    print("\n" + "=" * 50)
    print("IMMEDIATE ACTION REQUIRED")
    print("=" * 50)

    # High priority panels
    high_priority = latest_data[
        (latest_data['urgency_score'] > 40) |  # Lowered threshold
        (latest_data['efficiency'] < 85)
        ].sort_values('urgency_score', ascending=False)

    if len(high_priority) > 0:
        print(f"🚨 {len(high_priority)} panels require immediate attention:")
        for _, panel in high_priority.head(10).iterrows():
            print(f"  Panel {panel['panel_id']:6d} | Sector {panel['sector']:8s} | "
                  f"Efficiency: {panel['efficiency']:5.1f}% | "
                  f"Urgency: {panel['urgency_score']:5.1f}")
    else:
        print("✅ No panels require immediate attention.")

    print("\n" + "=" * 50)
    print("CLEANING SCHEDULE RECOMMENDATIONS")
    print("=" * 50)

    # Cleaning recommendations by urgency
    needs_cleaning = latest_data[latest_data['needs_cleaning'] == True].copy()
    if len(needs_cleaning) > 0:
        print(f"🧹 {len(needs_cleaning)} panels scheduled for cleaning:")

        # Group by sector for efficient cleaning routes
        sector_summary = needs_cleaning.groupby('sector').agg({
            'panel_id': 'count',
            'urgency_score': 'mean',
            'efficiency': 'mean'
        }).sort_values('urgency_score', ascending=False)

        print("\nCleaning Priority by Sector:")
        for sector, row in sector_summary.head(10).iterrows():
            print(f"  {sector:10s} | {int(row['panel_id']):3d} panels | "
                  f"Avg Efficiency: {row['efficiency']:5.1f}% | "
                  f"Priority Score: {row['urgency_score']:5.1f}")
    else:
        print("✅ No panels currently need cleaning.")

    print("\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS")
    print("=" * 50)

    print(f"Grid Performance Summary:")
    print(f"  • Average Efficiency: {latest_data['efficiency'].mean():.1f}%")
    print(f"  • Efficiency Range: {latest_data['efficiency'].min():.1f}% - {latest_data['efficiency'].max():.1f}%")
    print(f"  • Total Power Output: {latest_data['power_output'].sum() / 1000:.1f} MW")
    print(f"  • Estimated Daily Energy: {latest_data['power_output'].sum() * 8 / 1000:.1f} MWh")

    # Performance by location
    print(f"\nPerformance by Grid Location:")
    for i in range(0, int(np.sqrt(len(latest_data))), 2):
        for j in range(0, int(np.sqrt(len(latest_data))), 2):
            sector_data = latest_data[
                (latest_data['x_coord'] >= i) & (latest_data['x_coord'] < i + 2) &
                (latest_data['y_coord'] >= j) & (latest_data['y_coord'] < j + 2)
                ]
            if len(sector_data) > 0:
                avg_eff = sector_data['efficiency'].mean()
                status = "🟢" if avg_eff > 90 else "🟡" if avg_eff > 85 else "🔴"
                print(f"  Zone ({i}-{i + 2}, {j}-{j + 2}): {status} {avg_eff:.1f}%")

    print("\n" + "=" * 50)
    print("PREDICTIVE MODEL INSIGHTS")
    print("=" * 50)

    print("Top factors affecting cleaning needs:")
    feature_importance_sorted = sorted(zip(feature_names, feature_importance),
                                       key=lambda x: x[1], reverse=True)

    for i, (feature, importance) in enumerate(feature_importance_sorted[:8]):
        print(f"  {i + 1}. {feature:25s}: {importance:.3f}")

    print("\n" + "=" * 50)
    print("MAINTENANCE RECOMMENDATIONS")
    print("=" * 50)

    recommendations = []

    if latest_data['efficiency'].mean() < 85:
        recommendations.append("🔧 Grid efficiency is below optimal. Increase cleaning frequency.")

    if latest_data['dirt_coverage_percent'].mean() > 20:
        recommendations.append("🧹 High dirt accumulation detected. Consider weather protection measures.")

    hotspot_count = latest_data['hotspot_detected'].sum()
    if hotspot_count > 0:
        recommendations.append(f"⚠️  {hotspot_count} panels show hotspot activity. Schedule electrical inspection.")

    edge_panels_low_eff = latest_data[
        (latest_data['edge_factor'] < 0.5) & (latest_data['efficiency'] < 85)
        ]
    if len(edge_panels_low_eff) > 0:
        recommendations.append("🌪️  Edge panels showing reduced efficiency. Check for wind/dust damage.")

    if len(recommendations) == 0:
        recommendations.append("✅ Grid is operating within optimal parameters.")

    for rec in recommendations:
        print(f"  {rec}")

    print("\n" + "=" * 50)
    print("ESTIMATED COST IMPACT")
    print("=" * 50)

    # Simple cost estimates
    cleaning_cost_per_panel = 5  # USD
    lost_revenue_per_percent_per_day = 2  # USD per panel per efficiency percent lost per day

    cleaning_costs = len(needs_cleaning) * cleaning_cost_per_panel
    efficiency_loss = max(0, 95 - latest_data['efficiency'].mean())
    daily_revenue_loss = len(latest_data) * efficiency_loss * lost_revenue_per_percent_per_day

    print(f"  Immediate cleaning cost: ${cleaning_costs:,.2f}")
    print(f"  Daily revenue loss from efficiency: ${daily_revenue_loss:,.2f}")
    print(f"  ROI of cleaning (1 day): {(daily_revenue_loss / max(1, cleaning_costs) * 100):.1f}%")

    print("\n" + "=" * 80)


def main():
    """Main execution function"""
    print("Initializing Solar Panel Grid Maintenance System...")
    print("=" * 60)

    # Initialize the grid simulator
    grid_simulator = SolarPanelGridSimulator(grid_size_km=10, panels_per_km2=100)  # 10k panels for demo
    print(
        f"✅ Grid initialized: {grid_simulator.total_panels:,} panels across {grid_simulator.grid_size_km}×{grid_simulator.grid_size_km} km²")

    # Generate environmental and operational data
    print("🌍 Generating environmental and operational data...")
    data = grid_simulator.generate_environmental_data(days=30)
    print(f"✅ Generated {len(data):,} data points over 30 days")

    # Initialize and train the predictive models
    print("🤖 Training predictive maintenance models...")
    predictor = SolarMaintenancePredictor()
    performance = predictor.train_models(data)
    print("✅ Models trained successfully")

    # Make predictions
    print("🔮 Making maintenance predictions...")
    data_with_predictions, feature_importance = predictor.predict_maintenance_needs(data)

    # Get feature names for interpretation
    feature_names = [
        'temperature', 'humidity', 'wind_speed', 'surface_reflectivity',
        'dirt_coverage_percent', 'panel_discoloration', 'voltage', 'current',
        'days_since_last_cleaning', 'proximity_to_road', 'edge_factor',
        'has_rain', 'has_dust_storm', 'hotspot_detected'
    ]

    print("✅ Predictions completed")

    # Generate visualizations
    print("📊 Creating comprehensive visualizations...")
    visualize_results(data_with_predictions, grid_simulator)

    # Generate maintenance report
    print("📋 Generating maintenance report...")
    generate_maintenance_report(data_with_predictions, feature_importance, feature_names)

    print("\n" + "=" * 60)
    print("🎉 Solar Panel Grid Analysis Complete!")
    print("=" * 60)

    return data_with_predictions, predictor, grid_simulator


# Run the complete system
if __name__ == "__main__":
    data, predictor_model, grid_sim = main()
