import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from scipy.spatial import cKDTree

app = Flask(__name__)

# Load Dataset Directly
DATASET_PATH = 'Chennai_Geotechnical_Dataset_Completed_No_Missing_Data.xlsx'

print(f"Loading dataset from: {DATASET_PATH}...")
if os.path.exists(DATASET_PATH):
    df = pd.read_excel(DATASET_PATH)
    print("Dataset loaded successfully!")
else:
    raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' not found. Please ensure it is in your repository root.")

# Strict Coordinate Bounds as requested
LAT_MIN, LAT_MAX = 12.7817, 13.174
LNG_MIN, LNG_MAX = 80.094, 80.301

# Pre-calculate geographic center points for each dataset locality
locality_centers = df.groupby('Location (Chennai)')[['Latitude', 'Longitude']].mean().reset_index()

# Build Spatial KDTree across dataset points
coords = df[['Latitude', 'Longitude']].values
tree = cKDTree(coords)

ignore_cols = {'Borehole ID', 'Location (Chennai)', 'Latitude', 'Longitude', 'Soil Type', 'Recommended Foundation'}
numeric_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

def get_precise_locality(lat, lng):
    """Finds the dataset area whose center is geographically closest to input coordinates."""
    lats = locality_centers['Latitude'].values
    lngs = locality_centers['Longitude'].values
    dists = np.sqrt((lats - lat)**2 + (lngs - lng)**2)
    closest_idx = np.argmin(dists)
    return locality_centers.iloc[closest_idx]['Location (Chennai)']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error_msg = None

    if request.method == 'POST':
        try:
            user_lat = float(request.form.get('latitude'))
            user_lng = float(request.form.get('longitude'))

            # Bounds Validation
            if not (LAT_MIN <= user_lat <= LAT_MAX and LNG_MIN <= user_lng <= LNG_MAX):
                error_msg = "The given location is out of chennai"
            else:
                # 1. Map input coordinates directly to closest dataset locality area
                locality_name = get_precise_locality(user_lat, user_lng)

                # 2. Query spatial KDTree for nearest borehole samples
                distances, indices = tree.query([user_lat, user_lng], k=5)
                nearest_row = df.iloc[indices[0]]

                # Interpolate parameters
                if distances[0] < 1e-7:
                    pred_series = nearest_row[numeric_cols]
                else:
                    weights = 1.0 / (distances ** 2)
                    weights /= np.sum(weights)
                    pred_series = (df.iloc[indices][numeric_cols].values.T @ weights)

                # Prepare details (excluding Recommended Foundation temporarily to append last)
                details = {
                    'Locality Place': locality_name,
                    'Borehole Reference': nearest_row['Borehole ID'],
                    'Nearest Dataset Latitude': round(float(nearest_row['Latitude']), 6),
                    'Nearest Dataset Longitude': round(float(nearest_row['Longitude']), 6),
                    'Soil Type': nearest_row.get('Soil Type', 'N/A')
                }

                for i, col in enumerate(numeric_cols):
                    val = pred_series.iloc[i] if hasattr(pred_series, 'iloc') else pred_series[i]
                    if col in ['SPT N Value', 'Liquid Limit (%)', 'Plastic Limit (%)', 'Plasticity Index (%)', 'Cohesion (kPa)', 'Friction Angle (°)', 'Bearing Capacity (kN/m²)']:
                        details[col] = int(round(val))
                    elif 'Permeability' in col:
                        details[col] = f"{val:.4e}"
                    else:
                        details[col] = round(float(val), 2)

                # Ensure Recommended Foundation is added strictly at the end
                if 'Recommended Foundation' in df.columns:
                    details['Recommended Foundation'] = nearest_row['Recommended Foundation']

                result = {
                    'details': details,
                    'user_lat': user_lat,
                    'user_lng': user_lng
                }

        except ValueError:
            error_msg = "Please enter valid numerical values for latitude and longitude."

    return render_template('index.html', result=result, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
