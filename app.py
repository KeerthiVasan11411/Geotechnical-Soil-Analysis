from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load dataset once at startup
DATASET_PATH = 'Chennai_Geotechnical_Dataset_Completed_No_Missing_Data.xlsx'

if os.path.exists(DATASET_PATH):
    df = pd.read_excel(DATASET_PATH)
else:
    df = pd.DataFrame()

# Defined Chennai Bounds
LAT_MIN, LAT_MAX = 12.7836, 13.1693
LNG_MIN, LNG_MAX = 80.09551, 80.2977

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error_msg = None

    if request.method == 'POST':
        try:
            user_lat = float(request.form.get('latitude'))
            user_lng = float(request.form.get('longitude'))

            # Check if input is within valid Chennai bounds
            if not (LAT_MIN <= user_lat <= LAT_MAX and LNG_MIN <= user_lng <= LNG_MAX):
                error_msg = "The given location is out of chennai"
            else:
                if not df.empty:
                    # Calculate Euclidean distance to all data points
                    distances = np.sqrt(
                        (df['Latitude'] - user_lat)**2 + 
                        (df['Longitude'] - user_lng)**2
                    )
                    nearest_idx = distances.idxmin()
                    nearest_row = df.loc[nearest_idx].to_dict()
                    
                    # Store result details
                    result = {
                        'matched_lat': nearest_row.get('Latitude'),
                        'matched_lng': nearest_row.get('Longitude'),
                        'distance_km': round(distances[nearest_idx] * 111, 3), # Approximate conversion to km
                        'details': nearest_row
                    }
                else:
                    error_msg = "Dataset file not found or empty on server."

        except (ValueError, TypeError):
            error_msg = "Please enter valid numerical coordinates."

    return render_template('index.html', result=result, error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
