"""
Credit Card Approval Prediction — Flask Web Application
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

# ── Load model artifacts ───────────────────────────────────────────────────
MODEL_DIR = 'models'

def load_artifacts():
    artifacts = {}
    try:
        artifacts['model']          = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        artifacts['scaler']         = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        artifacts['encoders']       = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
        artifacts['feature_names']  = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        artifacts['model_name']     = joblib.load(os.path.join(MODEL_DIR, 'best_model_name.pkl'))
        print(f"[OK] Model loaded: {artifacts['model_name']}")
    except FileNotFoundError as e:
        print(f"[WARN]  Model not found: {e}")
        print("    Run  python model_training.py  first.")
        artifacts = None
    return artifacts

ARTIFACTS = load_artifacts()

# ── Dropdown option maps ───────────────────────────────────────────────────
OPTIONS = {
    'gender':         [('M', 'Male'), ('F', 'Female')],
    'own_car':        [('Y', 'Yes'), ('N', 'No')],
    'own_property':   [('Y', 'Yes'), ('N', 'No')],
    'income_type':    [
        ('Working',               'Working'),
        ('Commercial associate',  'Commercial Associate'),
        ('Pensioner',             'Pensioner'),
        ('State servant',         'State Servant'),
        ('Student',               'Student'),
    ],
    'education_type': [
        ('Secondary / secondary special', 'Secondary / Secondary Special'),
        ('Higher education',              'Higher Education'),
        ('Incomplete higher',             'Incomplete Higher'),
        ('Lower secondary',               'Lower Secondary'),
        ('Academic degree',               'Academic Degree'),
    ],
    'family_status':  [
        ('Married',              'Married'),
        ('Single / not married', 'Single / Not Married'),
        ('Civil marriage',       'Civil Marriage'),
        ('Separated',            'Separated'),
        ('Widow',                'Widow'),
    ],
    'housing_type':   [
        ('House / apartment',   'House / Apartment'),
        ('With parents',        'With Parents'),
        ('Municipal apartment', 'Municipal Apartment'),
        ('Rented apartment',    'Rented Apartment'),
        ('Office apartment',    'Office Apartment'),
        ('Co-op apartment',     'Co-op Apartment'),
    ],
    'occupation_type': [
        ('Laborers',             'Laborers'),
        ('Core staff',           'Core Staff'),
        ('Accountants',          'Accountants'),
        ('Managers',             'Managers'),
        ('Drivers',              'Drivers'),
        ('Sales staff',          'Sales Staff'),
        ('Cleaning staff',       'Cleaning Staff'),
        ('Cooking staff',        'Cooking Staff'),
        ('Private service staff','Private Service Staff'),
        ('Medicine staff',       'Medicine Staff'),
        ('Security staff',       'Security Staff'),
        ('High skill tech staff','High Skill Tech Staff'),
        ('IT staff',             'IT Staff'),
        ('Secretaries',          'Secretaries'),
    ],
}


# ── Feature engineering (mirrors model_training.py) ────────────────────────
def engineer_and_encode(form_data: dict, artifacts: dict) -> np.ndarray:
    encoders = artifacts['encoders']

    gender         = form_data.get('gender', 'M')
    own_car        = form_data.get('own_car', 'N')
    own_property   = form_data.get('own_property', 'N')
    children_cnt   = int(form_data.get('children_count', 0))
    annual_income  = float(form_data.get('annual_income', 100000))
    income_type    = form_data.get('income_type', 'Working')
    education_type = form_data.get('education_type', 'Secondary / secondary special')
    family_status  = form_data.get('family_status', 'Married')
    housing_type   = form_data.get('housing_type', 'House / apartment')
    occupation_type= form_data.get('occupation_type', 'Laborers')
    family_members = int(form_data.get('family_members', 2))
    has_work_phone = int(form_data.get('has_work_phone', 0))
    has_phone      = int(form_data.get('has_phone', 0))
    has_email      = int(form_data.get('has_email', 0))
    age_years      = float(form_data.get('age_years', 0))
    employment_years = float(form_data.get('employment_years', 3))

    # Derived
    income_per_family = annual_income / (family_members + 1)
    is_employed       = 1 if employment_years > 0 else 0
    high_income       = 1 if annual_income > 200000 else 0

    # Encode categoricals
    def safe_encode(le, val):
        try:
            return le.transform([val])[0]
        except ValueError:
            return le.transform([le.classes_[0]])[0]

    row = [
        safe_encode(encoders['Gender'],          gender),
        safe_encode(encoders['Own_Car'],         own_car),
        safe_encode(encoders['Own_Property'],    own_property),
        children_cnt,
        annual_income,
        safe_encode(encoders['Income_Type'],     income_type),
        safe_encode(encoders['Education_Type'],  education_type),
        safe_encode(encoders['Family_Status'],   family_status),
        safe_encode(encoders['Housing_Type'],    housing_type),
        has_work_phone,
        has_phone,
        has_email,
        safe_encode(encoders['Occupation_Type'], occupation_type),
        family_members,
        age_years,
        employment_years,
        income_per_family,
        is_employed,
        high_income,
    ]

    # Align to training feature order
    feature_names = artifacts['feature_names']
    col_map = {
        'Gender':           row[0],
        'Own_Car':          row[1],
        'Own_Property':     row[2],
        'Children_Count':   row[3],
        'Annual_Income':    row[4],
        'Income_Type':      row[5],
        'Education_Type':   row[6],
        'Family_Status':    row[7],
        'Housing_Type':     row[8],
        'Has_Work_Phone':   row[9],
        'Has_Phone':        row[10],
        'Has_Email':        row[11],
        'Occupation_Type':  row[12],
        'Family_Members':   row[13],
        'Age_Years':        row[14],
        'Employment_Years': row[15],
        'Income_Per_Family':row[16],
        'Is_Employed':      row[17],
        'High_Income':      row[18],
    }
    aligned = np.array([col_map.get(f, 0) for f in feature_names], dtype=float)
    return aligned


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html', options=OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    if ARTIFACTS is None:
        return render_template('result.html',
                               error="Model not loaded. Please train first.",
                               options=OPTIONS)

    form_data = request.form.to_dict()
    features  = engineer_and_encode(form_data, ARTIFACTS)
    features_s = ARTIFACTS['scaler'].transform(features.reshape(1, -1))

    prediction = ARTIFACTS['model'].predict(features_s)[0]
    probability = ARTIFACTS['model'].predict_proba(features_s)[0]

    result = {
        'prediction':    int(prediction),
        'label':         'Approved' if prediction == 1 else 'Rejected',
        'confidence':    float(max(probability)) * 100,
        'approve_prob':  float(probability[1]) * 100,
        'reject_prob':   float(probability[0]) * 100,
        'model_name':    ARTIFACTS['model_name'],
        'form_data':     form_data,
    }

    # Feature importance (if available)
    model = ARTIFACTS['model']
    feature_names = ARTIFACTS['feature_names']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:6]
        result['top_features'] = [
            {'name': feature_names[i].replace('_', ' '), 'value': round(importances[i] * 100, 2)}
            for i in top_idx
        ]
    else:
        result['top_features'] = []

    return render_template('result.html', result=result, options=OPTIONS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access and Watson integration."""
    if ARTIFACTS is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json(force=True)
    features  = engineer_and_encode(data, ARTIFACTS)
    features_s = ARTIFACTS['scaler'].transform(features.reshape(1, -1))

    prediction  = ARTIFACTS['model'].predict(features_s)[0]
    probability = ARTIFACTS['model'].predict_proba(features_s)[0]

    return jsonify({
        'prediction':   int(prediction),
        'label':        'Approved' if prediction == 1 else 'Rejected',
        'confidence':   float(max(probability)),
        'approve_prob': float(probability[1]),
        'reject_prob':  float(probability[0]),
        'model':        ARTIFACTS['model_name'],
    })


@app.route('/about')
def about():
    model_name = ARTIFACTS['model_name'] if ARTIFACTS else "Not loaded"
    return render_template('index.html', options=OPTIONS,
                           model_name=model_name, show_about=True)


@app.route('/health')
def health():
    return jsonify({
        'status':     'ok',
        'model_loaded': ARTIFACTS is not None,
        'model_name': ARTIFACTS['model_name'] if ARTIFACTS else None,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
