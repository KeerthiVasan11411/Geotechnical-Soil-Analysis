import os
import pandas as pd
from flask import Flask, jsonify, render_template, request
from scipy.spatial import cKDTree

app = Flask(__name__)

# File name matching your uploaded GitHub repository file
DATASET_PATH = (
    "Chennai_Geotechnical_Dataset_500000_CORRECT_UNIQUE_COORDINATES.xlsx"
)

# Load dataset
print(f"Loading dataset from: {DATASET_PATH}...")
if os.path.exists(DATASET_PATH):
    df = pd.read_excel(DATASET_PATH)
    print("Dataset loaded successfully!")
else:
    raise FileNotFoundError(f"Dataset file '{DATASET_PATH}' not found.")

# Pre-build spatial tree for fast coordinate lookup
coords = df[["Latitude", "Longitude"]].values
tree = cKDTree(coords)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))

        # Find nearest coordinate match in the spatial dataset
        distance, index = tree.query([lat, lon])
        result = df.iloc[index].to_dict()

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
