"""
Credit Card Approval Prediction — ML Training Pipeline
Trains 4 classifiers, evaluates, and saves the best model.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.tree               import DecisionTreeClassifier
from sklearn.metrics            import (accuracy_score, precision_score,
                                        recall_score, f1_score,
                                        roc_auc_score, confusion_matrix,
                                        classification_report, roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join('data', 'credit_card_data.csv')
MODEL_DIR   = 'models'
PLOTS_DIR   = os.path.join('static', 'images', 'plots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Plot styling ───────────────────────────────────────────────────────────
DARK_BG   = '#0a0e27'
CARD_BG   = '#12183a'
VIOLET    = '#6c63ff'
CYAN      = '#00d4ff'
GOLD      = '#ffd700'
RED_SOFT  = '#ff6b8a'
GREEN_S   = '#00e5a0'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   CARD_BG,
    'axes.edgecolor':   '#2a3560',
    'text.color':       '#e0e6ff',
    'axes.labelcolor':  '#e0e6ff',
    'xtick.color':      '#8892b0',
    'ytick.color':      '#8892b0',
    'grid.color':       '#1e2d5a',
    'font.family':      'DejaVu Sans',
})


# ══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

def load_and_engineer(path: str) -> tuple[pd.DataFrame, pd.Series]:
    print("\n[1/5] Loading and engineering features …")
    df = pd.read_csv(path)
    print(f"      Raw shape : {df.shape}")

    # ── Derived features ───────────────────────────────────────────────
    df['Age_Years']        = np.abs(df['Age_Days']) / 365.25
    df['Employment_Years'] = np.where(
        df['Employment_Days'] > 0,
        0,
        np.abs(df['Employment_Days']) / 365.25)
    df['Income_Per_Family'] = df['Annual_Income'] / (df['Family_Members'] + 1)
    df['Is_Employed']       = (df['Employment_Days'] < 0).astype(int)
    df['High_Income']       = (df['Annual_Income'] > 200000).astype(int)

    # ── Drop raw day columns ───────────────────────────────────────────
    df.drop(columns=['Age_Days', 'Employment_Days'], inplace=True)

    # ── Encode categoricals ────────────────────────────────────────────
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    print(f"      Encoders saved for: {cat_cols}")

    X = df.drop(columns=['Status'])
    y = df['Status']
    print(f"      Final features : {X.shape[1]}")
    print(f"      Class balance  : {y.value_counts().to_dict()}")
    return X, y, encoders


# ══════════════════════════════════════════════════════════════════════════
# 2.  TRAIN / TEST SPLIT + SCALING
# ══════════════════════════════════════════════════════════════════════════

def split_and_scale(X, y):
    print("\n[2/5] Splitting and scaling data …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # Save feature names
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, 'feature_names.pkl'))

    print(f"      Train : {X_train_s.shape}  |  Test : {X_test_s.shape}")
    return X_train_s, X_test_s, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════
# 3.  TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════

def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            n_jobs=-1, random_state=42),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    print("\n[3/5] Training and evaluating models …")
    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n  >> {name}")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring='accuracy', n_jobs=-1)
        metrics = {
            'model':       model,
            'accuracy':    accuracy_score(y_test, y_pred),
            'precision':   precision_score(y_test, y_pred),
            'recall':      recall_score(y_test, y_pred),
            'f1':          f1_score(y_test, y_pred),
            'roc_auc':     roc_auc_score(y_test, y_proba),
            'cv_mean':     cv_scores.mean(),
            'cv_std':      cv_scores.std(),
            'y_pred':      y_pred,
            'y_proba':     y_proba,
        }
        results[name] = metrics
        print(f"     Accuracy  : {metrics['accuracy']:.4f}")
        print(f"     F1-Score  : {metrics['f1']:.4f}")
        print(f"     ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"     CV        : {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# 4.  SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════

def save_best(results):
    print("\n[4/5] Selecting and saving best model …")
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best      = results[best_name]
    joblib.dump(best['model'], os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(best_name,     os.path.join(MODEL_DIR, 'best_model_name.pkl'))
    print(f"      Best model : {best_name}")
    print(f"      ROC-AUC    : {best['roc_auc']:.4f}")
    print(f"      Saved -> {MODEL_DIR}/best_model.pkl")
    return best_name


# ══════════════════════════════════════════════════════════════════════════
# 5.  GENERATE PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_metrics_comparison(results):
    names   = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors  = [VIOLET, CYAN, GOLD, RED_SOFT, GREEN_S]

    x     = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor='none')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7.5, color='#e0e6ff')

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold',
                 color=CYAN, pad=18)
    ax.legend(loc='upper right', framealpha=0.3, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'metrics_comparison.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved -> {path}")


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    palette = [VIOLET, CYAN, GOLD, RED_SOFT]
    for (name, res), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC={res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], '--', color='#8892b0', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=16, fontweight='bold',
                 color=CYAN, pad=18)
    ax.legend(loc='lower right', framealpha=0.3, fontsize=10)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved -> {path}")


def plot_confusion_matrix(results, y_test, best_name):
    cm   = confusion_matrix(y_test, results[best_name]['y_pred'])
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)

    cmap = sns.diverging_palette(240, 280, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'],
                linewidths=2, linecolor=DARK_BG,
                annot_kws={'size': 18, 'weight': 'bold', 'color': 'white'},
                ax=ax)

    ax.set_xlabel('Predicted', fontsize=13, labelpad=10)
    ax.set_ylabel('Actual',    fontsize=13, labelpad=10)
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=15,
                 fontweight='bold', color=CYAN, pad=16)
    ax.set_facecolor(CARD_BG)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved -> {path}")


def plot_feature_importance(results, feature_names, best_name):
    model = results[best_name]['model']
    if not hasattr(model, 'feature_importances_'):
        print(f"      {best_name} has no feature_importances_, skipping.")
        return

    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in idx]
    top_vals     = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    bar_colors = [VIOLET if v > np.median(top_vals) else CYAN for v in top_vals]
    bars = ax.barh(range(len(top_features)), top_vals[::-1],
                   color=bar_colors[::-1], edgecolor='none', height=0.65)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=11)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top Features — {best_name}', fontsize=16,
                 fontweight='bold', color=CYAN, pad=18)
    ax.grid(axis='x', alpha=0.25)

    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color='#e0e6ff')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved -> {path}")


def plot_cv_scores(results):
    names    = list(results.keys())
    cv_means = [results[n]['cv_mean'] for n in names]
    cv_stds  = [results[n]['cv_std']  for n in names]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    colors = [VIOLET, CYAN, GOLD, GREEN_S]
    bars   = ax.bar(names, cv_means, color=colors, edgecolor='none',
                    alpha=0.85, width=0.55,
                    yerr=cv_stds, capsize=6, error_kw={'color': '#ffffff', 'lw': 2})

    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.005,
                f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10, color='#e0e6ff')

    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax.set_title('5-Fold Cross-Validation Scores', fontsize=16,
                 fontweight='bold', color=CYAN, pad=18)
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'cv_scores.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Credit Card Approval Prediction — Training Pipeline")
    print("=" * 60)

    # Generate dataset if not present
    if not os.path.exists(DATA_PATH):
        print("\nDataset not found — generating …")
        import subprocess
        subprocess.run([sys.executable, os.path.join('data', 'generate_dataset.py')],
                       check=True)

    X, y, encoders = load_and_engineer(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    best_name = save_best(results)

    print("\n[5/5] Generating visualisation plots …")
    feature_names = list(
        joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl')))
    plot_metrics_comparison(results)
    plot_roc_curves(results, y_test)
    plot_confusion_matrix(results, y_test, best_name)
    plot_feature_importance(results, feature_names, best_name)
    plot_cv_scores(results)

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<25} {'Acc':>7} {'F1':>7} {'AUC':>7} {'CV':>7}")
    print("-" * 60)
    for name, res in results.items():
        marker = " *" if name == best_name else ""
        print(f"{name:<25} {res['accuracy']:>7.4f} {res['f1']:>7.4f}"
              f" {res['roc_auc']:>7.4f} {res['cv_mean']:>7.4f}{marker}")
    print("=" * 60)
    print(f"\n[OK] Best model: {best_name}")
    print(f"   All artifacts saved to ./{MODEL_DIR}/")
    print(f"   All plots saved to    ./{PLOTS_DIR}/")


if __name__ == '__main__':
    main()
