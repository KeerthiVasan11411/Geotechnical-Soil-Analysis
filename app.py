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

# Coordinate Bounds covering Greater Chennai Metropolitan Area
LAT_MIN, LAT_MAX = 12.7500, 13.2500
LNG_MIN, LNG_MAX = 80.0000, 80.3500

# Pre-calculate geographic center points for each dataset locality
locality_centers = df.groupby('Location (Chennai)')[['Latitude', 'Longitude']].mean().reset_index()

# Build Spatial KDTree across dataset points
coords = df[['Latitude', 'Longitude']].values
tree = cKDTree(coords)

ignore_cols = {'Borehole ID', 'Location (Chennai)', 'Latitude', 'Longitude', 'Soil Type', 'Recommended Foundation'}
numeric_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

import json
import urllib.request

LOCALITY_PINCODES = {
    'Adyar': '600020',
    'Alandur': '600016',
    'Ambattur': '600053',
    'Anna Nagar': '600040',
    'Avadi': '600054',
    'Besant Nagar': '600090',
    'Chromepet': '600044',
    'Guindy': '600032',
    'Kelambakkam': '603103',
    'Kodambakkam': '600024',
    'Koyambedu': '600107',
    'Medavakkam': '600100',
    'Mogappair': '600037',
    'Mylapore': '600004',
    'Navalur': '603103',
    'Nungambakkam': '600034',
    'Pallavaram': '600043',
    'Pallikaranai': '600100',
    'Perambur': '600011',
    'Perungudi': '600096',
    'Poonamallee': '600056',
    'Porur': '600116',
    'Puzhal': '600066',
    'Royapuram': '600013',
    'Sholinganallur': '600119',
    'T Nagar': '600017',
    'Tambaram': '600045',
    'Thiruvanmiyur': '600041',
    'Tondiarpet': '600081',
    'Velachery': '600042'
}

def get_area_and_pincode(lat, lng, locality_name):
    """Returns strictly ONLY the Area Name and Pincode (e.g. 'Guindy - 600032')."""
    area = locality_name
    pincode = LOCALITY_PINCODES.get(locality_name, '')
    
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=18&addressdetails=1"
        headers = {'User-Agent': 'ChennaiSoilAnalysisApp/1.0 (contact: support@chennaisoil.local)'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            addr = data.get('address', {})
            
            # Prefer specific neighbourhood or suburb if available
            for key in ['neighbourhood', 'suburb', 'residential', 'city_district']:
                val = addr.get(key, '').strip()
                if val and not val.startswith('Ward ') and not val.startswith('Zone ') and val.lower() not in ['chennai', 'chennai corporation']:
                    area = val
                    break
            
            if addr.get('postcode'):
                pincode = addr.get('postcode').strip()
    except Exception:
        pass

    if pincode:
        return f"{area} - {pincode}"
    return area

def get_precise_locality(lat, lng):
    """Finds the dataset area whose nearest borehole point is closest."""
    dist, idx = tree.query([lat, lng])
    return df.iloc[idx]['Location (Chennai)']

@app.route('/api/geocode', methods=['GET'])
def api_geocode():
    """Live JSON endpoint for real-time interactive map picking."""
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        
        dist, idx = tree.query([lat, lng])
        nearest_row = df.iloc[idx]
        locality_name = nearest_row['Location (Chennai)']
        area_pin = get_area_and_pincode(lat, lng, locality_name)
        
        return {
            'status': 'success',
            'locality': locality_name,
            'area_pincode': area_pin,
            'borehole_id': str(nearest_row['Borehole ID']),
            'soil_type': str(nearest_row.get('Soil Type', 'N/A'))
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 400

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
                # 1. Query spatial KDTree for nearest borehole samples
                distances, indices = tree.query([user_lat, user_lng], k=5)
                nearest_row = df.iloc[indices[0]]

                # Exact dataset locality from closest borehole sample
                locality_name = nearest_row.get('Location (Chennai)', get_precise_locality(user_lat, user_lng))

                # Strictly Area Name and Pincode only
                area_with_pin = get_area_and_pincode(user_lat, user_lng, locality_name)

                # Interpolate parameters
                if distances[0] < 1e-7:
                    pred_series = nearest_row[numeric_cols]
                else:
                    weights = 1.0 / (distances ** 2)
                    weights /= np.sum(weights)
                    pred_series = (df.iloc[indices][numeric_cols].values.T @ weights)

                # Prepare details (excluding Recommended Foundation temporarily to append last)
                details = {
                    'Area & Pincode': area_with_pin,
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
                    'user_lat': round(user_lat, 6),
                    'user_lng': round(user_lng, 6)
                }

        except ValueError:
            error_msg = "Please enter valid numerical values for latitude and longitude."

    return render_template('index.html', result=result, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
