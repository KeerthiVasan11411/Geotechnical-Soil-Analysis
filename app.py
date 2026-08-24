from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Chennai Geotechnical Dataset
df = pd.read_excel("Chennai_Geotechnical_Dataset.xlsx")

# Define Chennai Bounding Box Limits
LAT_MIN, LAT_MAX = 12.7817, 13.1740
LON_MIN, LON_MAX = 80.0940, 80.3010

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error_message = None

    if request.method == 'POST':
        try:
            user_lat = float(request.form['latitude'])
            user_lon = float(request.form['longitude'])

            # Check if input is within Chennai boundary
            if not (LAT_MIN <= user_lat <= LAT_MAX and LON_MIN <= user_lon <= LON_MAX):
                error_message = "The given location is out of the chennai"
            else:
                # Calculate Euclidean distance to find nearest borehole record
                distances = np.sqrt((df['Latitude'] - user_lat)**2 + (df['Longitude'] - user_lon)**2)
                nearest_idx = distances.idxmin()
                
                # Extract soil details as dictionary
                result = df.loc[nearest_idx].fillna("N/A").to_dict()

        except Exception as e:
            error_message = f"Invalid input format: {e}"

    return render_template('index.html', result=result, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)