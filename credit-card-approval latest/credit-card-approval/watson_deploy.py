"""
Credit Card Approval Prediction — IBM Watson ML Deployment Pipeline
Deploys the best trained model to IBM Watson Machine Learning for cloud inference.
"""

import os
import json
import joblib
import numpy as np

# ── Watson ML credentials (fill in your IBM Cloud credentials) ─────────────
WML_CREDENTIALS = {
    "url":    os.getenv("WML_URL", "https://us-south.ml.cloud.ibm.com"),
    "apikey": os.getenv("WML_APIKEY", "YOUR_IBM_CLOUD_API_KEY_HERE"),
}
SPACE_ID = os.getenv("WML_SPACE_ID", "YOUR_DEPLOYMENT_SPACE_ID_HERE")


def deploy_model():
    """Deploy the saved best model to IBM Watson ML."""
    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        print("ibm-watson-machine-learning not installed. Run: pip install ibm-watson-machine-learning")
        return None

    print("Connecting to IBM Watson Machine Learning …")
    client = APIClient(WML_CREDENTIALS)
    client.set.default_space(SPACE_ID)

    # ── Store the model ────────────────────────────────────────────────────
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found. Run model_training.py first.")
        return None

    model_meta = {
        client.repository.ModelMetaNames.NAME:            "Credit Card Approval Model",
        client.repository.ModelMetaNames.TYPE:            "scikit-learn_1.3",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: client.software_specifications.get_uid_by_name(
            "runtime-23.1-py3.10"),
    }

    print("Storing model in Watson ML repository …")
    stored_model = client.repository.store_model(
        model_path,
        meta_props=model_meta,
    )
    model_uid = stored_model["metadata"]["id"]
    print(f"Model stored → UID: {model_uid}")

    # ── Create online deployment ───────────────────────────────────────────
    deploy_meta = {
        client.deployments.ConfigurationMetaNames.NAME:            "Credit Card Approval Endpoint",
        client.deployments.ConfigurationMetaNames.ONLINE:          {},
        client.deployments.ConfigurationMetaNames.SERVING_NAME:    "credit_card_approval",
    }

    print("Creating online deployment …")
    deployment = client.deployments.create(model_uid, meta_props=deploy_meta)
    deployment_uid = deployment["metadata"]["id"]
    scoring_url    = client.deployments.get_scoring_href(deployment)

    print(f"\n✅ Deployment successful!")
    print(f"   Deployment UID : {deployment_uid}")
    print(f"   Scoring URL    : {scoring_url}")

    # Save deployment info locally
    deploy_info = {
        "model_uid":      model_uid,
        "deployment_uid": deployment_uid,
        "scoring_url":    scoring_url,
    }
    os.makedirs("models", exist_ok=True)
    with open("models/watson_deployment.json", "w") as f:
        json.dump(deploy_info, f, indent=2)
    print("   Deployment info saved → models/watson_deployment.json")

    return deployment_uid, scoring_url


def score_watson(scoring_url, feature_values: list) -> dict:
    """
    Send a scoring request to the Watson ML deployment.

    Parameters
    ----------
    scoring_url   : str  — endpoint URL from deploy_model()
    feature_values: list — preprocessed feature array (same order as training)

    Returns
    -------
    dict with 'prediction' (0/1) and 'probability' ([p_reject, p_approve])
    """
    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        return {"error": "ibm-watson-machine-learning not installed"}

    client = APIClient(WML_CREDENTIALS)
    client.set.default_space(SPACE_ID)

    payload = {
        "input_data": [{
            "values": [feature_values],
        }]
    }

    response   = client.deployments.score(scoring_url, payload)
    prediction = response["predictions"][0]["values"][0][0]
    probability = response["predictions"][0]["values"][0][1]

    return {
        "prediction":  prediction,
        "probability": probability,
        "label":       "Approved" if prediction == 1 else "Rejected",
    }


def list_deployments():
    """List all active Watson ML deployments."""
    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        print("ibm-watson-machine-learning not installed.")
        return

    client = APIClient(WML_CREDENTIALS)
    client.set.default_space(SPACE_ID)
    client.deployments.list()


if __name__ == "__main__":
    print("=" * 60)
    print("  IBM Watson Machine Learning — Deployment Pipeline")
    print("=" * 60)
    print("\nNOTE: Set environment variables before running:")
    print("  WML_URL      — your Watson ML instance URL")
    print("  WML_APIKEY   — your IBM Cloud API key")
    print("  WML_SPACE_ID — your deployment space ID")
    print()

    result = deploy_model()
    if result:
        uid, url = result
        print(f"\nTest scoring request …")
        # Example feature array (16 features after encoding)
        example = [0, 1, 1, 0, 250000, 1, 2, 0, 0, -2000, -14000, 0, 1, 0, 5, 2]
        score   = score_watson(url, example)
        print(f"Result: {score}")
