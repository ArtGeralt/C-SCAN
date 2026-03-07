"""
small_data_qsar.py
==================
QSAR model development toolkit for small datasets (< 100-200 compounds).

Part 1 of new_development.txt:
  - Baseline regressors and classifiers: PLS, SVM, RF, XGBoost
  - Feature selection: variance filter, correlation filter, RFE, tree-based importance
  - Cross-validation: LOOCV, Repeated Stratified K-Fold, Y-Randomization
  - Evaluation metrics: R², Q², RMSE (regression); ROC-AUC, F1 (classification)

Usage
-----
    # Regression (pIC50)
    from small_data_qsar import SmallDataQSAR
    qsar = SmallDataQSAR(task='regression', n_features_to_select=20)
    results = qsar.fit_evaluate("data/processed/20260130/kit_descriptors.csv",
                                 target_col='pIC50',
                                 output_dir="models/small_data/regression")

    # Classification (active/inactive)
    qsar = SmallDataQSAR(task='classification', n_features_to_select=20)
    results = qsar.fit_evaluate("data/processed/20260130/kit_descriptors_with_class.csv",
                                 target_col='activity_class',
                                 output_dir="models/small_data/classification")
"""

from __future__ import annotations

import os
import pickle
import warnings
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

timestamp = datetime.now().strftime("%Y%m%d")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_EXCLUDE_COLS = [
    "molecule_chembl_id", "canonical_smiles", "Smiles",
    "pIC50", "activity_class", "Activity_Level",
    "IC50_pActivity", "Kd_pActivity", "Ki_pActivity", "Inhibition_percent",
]


def _load_xy(csv_path: str, target_col: str):
    """Load a CSV and split into feature matrix X and target y."""
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. "
                         f"Available: {df.columns.tolist()}")
    exclude = [c for c in _EXCLUDE_COLS if c in df.columns]
    if target_col not in exclude:
        exclude.append(target_col)
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, df[target_col].values, feature_cols, df


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

class FeatureSelector:
    """
    Three-stage feature selector:
      1. Variance threshold (remove near-zero-variance descriptors)
      2. Pairwise correlation filter (remove one of each highly correlated pair)
      3. Recursive Feature Elimination (RFE) with a base estimator
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.9,
        n_features: int | None = None,
        rfe_estimator: str = "rf",
        task: str = "regression",
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features = n_features
        self.rfe_estimator = rfe_estimator
        self.task = task

        self._var_selector: VarianceThreshold | None = None
        self._corr_mask: np.ndarray | None = None
        self._rfe: RFE | None = None
        self.selected_indices_: np.ndarray | None = None
        self.selected_names_: list[str] = []

    # ------------------------------------------------------------------
    def _make_rfe_estimator(self):
        if self.rfe_estimator == "rf":
            return (
                RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                if self.task == "regression"
                else RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            )
        elif self.rfe_estimator == "svm":
            return SVR(kernel="linear") if self.task == "regression" else SVC(kernel="linear")
        raise ValueError(f"Unknown rfe_estimator: {self.rfe_estimator}")

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        n_original = X.shape[1]

        # Stage 1 – variance filter
        self._var_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self._var_selector.fit_transform(X)
        var_names = [feature_names[i] for i in self._var_selector.get_support(indices=True)]
        print(f"  [Feature selection] Variance filter: {n_original} -> {X_var.shape[1]} features")

        # Stage 2 – correlation filter
        corr_matrix = np.corrcoef(X_var.T)
        upper = np.triu(np.abs(corr_matrix), k=1)
        drop_idx = set()
        for col in range(upper.shape[1]):
            if any(upper[:col, col] > self.correlation_threshold):
                drop_idx.add(col)
        keep_mask = np.array([i not in drop_idx for i in range(X_var.shape[1])])
        self._corr_mask = keep_mask
        X_corr = X_var[:, keep_mask]
        corr_names = [var_names[i] for i, k in enumerate(keep_mask) if k]
        print(f"  [Feature selection] Correlation filter: {X_var.shape[1]} -> {X_corr.shape[1]} features")

        # Stage 3 – RFE
        n_sel = self.n_features or max(5, X_corr.shape[1] // 3)
        n_sel = min(n_sel, X_corr.shape[1])
        estimator = self._make_rfe_estimator()
        self._rfe = RFE(estimator=estimator, n_features_to_select=n_sel, step=0.1)
        self._rfe.fit(X_corr, y)
        X_rfe = np.asarray(self._rfe.transform(X_corr))
        self.selected_names_ = [corr_names[i] for i in self._rfe.get_support(indices=True)]
        print(f"  [Feature selection] RFE: {X_corr.shape[1]} -> {X_rfe.shape[1]} features")

        return X_rfe

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._var_selector is not None, "Call fit() first."
        assert self._corr_mask is not None
        assert self._rfe is not None
        X_var = self._var_selector.transform(X)
        X_corr = X_var[:, self._corr_mask]
        return np.asarray(self._rfe.transform(X_corr))

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        return self.fit(X, y, feature_names)


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def loocv_score(model, X: np.ndarray, y: np.ndarray, task: str = "regression") -> dict:
    """Leave-One-Out Cross-Validation.

    Returns Q²/accuracy plus raw CV predictions.
    """
    loo = LeaveOneOut()
    y_cv = cross_val_predict(model, X, y, cv=loo)

    if task == "regression":
        ss_res = float(np.sum((y - y_cv) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        q2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(mean_squared_error(y, y_cv)))
        print(f"  LOOCV  Q² = {q2:.3f}   RMSE = {rmse:.3f}")
        return {"q2": q2, "rmse_cv": rmse, "y_cv": y_cv}
    else:
        y_enc = LabelEncoder().fit_transform(y) if y.dtype == object else y
        y_cv_enc = LabelEncoder().fit_transform(y_cv) if y_cv.dtype == object else y_cv  # type: ignore[arg-type]
        acc = float(accuracy_score(y_enc, y_cv_enc))
        f1 = float(f1_score(y_enc, y_cv_enc, average="weighted"))
        print(f"  LOOCV  Accuracy = {acc:.3f}   F1 = {f1:.3f}")
        return {"accuracy_cv": acc, "f1_cv": f1, "y_cv": y_cv}


def repeated_kfold_score(
    model, X: np.ndarray, y: np.ndarray,
    n_splits: int = 5, n_repeats: int = 10,
    task: str = "regression",
) -> dict:
    """Repeated Stratified K-Fold CV (classification) or Repeated K-Fold (regression)."""
    if task == "classification":
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        scoring = "f1_weighted"
    else:
        from sklearn.model_selection import RepeatedKFold
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)  # type: ignore[assignment]
        scoring = "r2"

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    mean_s, std_s = float(scores.mean()), float(scores.std())
    label = "F1" if task == "classification" else "R²"
    print(f"  Repeated {n_splits}-Fold ({n_repeats}x)  {label} = {mean_s:.3f} ± {std_s:.3f}")
    return {"mean": mean_s, "std": std_s, "scores": scores.tolist()}


def y_randomization_test(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 100,
    task: str = "regression",
    cv: int = 5,
) -> dict:
    """Y-Randomization (target shuffling) test.

    Retrains the model with shuffled targets *n_trials* times. If the
    randomized models achieve performance comparable to the true model, the
    original model is exploiting chance correlations.

    Returns a dict with randomized scores and a pass/fail verdict.
    """
    rng = np.random.default_rng(42)
    scoring = "r2" if task == "regression" else "f1_weighted"
    rand_scores = []

    for _ in range(n_trials):
        y_shuffled = rng.permutation(y)
        s = cross_val_score(model, X, y_shuffled, cv=cv, scoring=scoring, n_jobs=-1).mean()
        rand_scores.append(float(s))

    # True model score
    true_score = float(cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean())
    rand_mean = float(np.mean(rand_scores))
    rand_std  = float(np.std(rand_scores))

    # Verdict: true model must be > mean + 2*std of randomized
    passed = true_score > (rand_mean + 2 * rand_std)
    verdict = "PASS" if passed else "FAIL (possible chance correlation)"
    metric_label = "R²" if task == "regression" else "F1"

    print(f"\n  Y-Randomization test ({n_trials} trials):")
    print(f"    True model {metric_label}         = {true_score:.3f}")
    print(f"    Randomized {metric_label} (mean)  = {rand_mean:.3f} ± {rand_std:.3f}")
    print(f"    Verdict: {verdict}")

    return {
        "true_score": true_score,
        "rand_mean": rand_mean,
        "rand_std": rand_std,
        "rand_scores": rand_scores,
        "passed": passed,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_pls(X_train, y_train, max_components: int = 10) -> PLSRegression:
    """Fit PLS regression, selecting n_components by LOOCV Q²."""
    best_q2, best_n = -np.inf, 2
    n_max = min(max_components, X_train.shape[0] - 1, X_train.shape[1])
    for n in range(1, n_max + 1):
        pls = PLSRegression(n_components=n, max_iter=1000)
        loo = LeaveOneOut()
        y_cv = cross_val_predict(pls, X_train, y_train, cv=loo)
        ss_res = float(np.sum((y_train - y_cv) ** 2))
        ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
        q2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if q2 > best_q2:
            best_q2, best_n = q2, n
    print(f"  PLS selected n_components={best_n}  (LOOCV Q²={best_q2:.3f})")
    model = PLSRegression(n_components=best_n, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def _build_svm(X_train, y_train, task: str = "regression", optimize: bool = True):
    """Fit SVM with RBF kernel, optionally grid-searching C and gamma."""
    if task == "regression":
        BaseCls = SVR
        param_grid: dict[str, list] = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.001]}
        scoring = "r2"

        if optimize:
            gs = GridSearchCV(BaseCls(kernel="rbf"), param_grid,
                              cv=min(5, len(y_train)), scoring=scoring, n_jobs=-1)
            gs.fit(X_train, y_train)
            print(f"  SVM best params: {gs.best_params_}")
            return gs.best_estimator_
        m = BaseCls(kernel="rbf")
        m.fit(X_train, y_train)
        return m
    else:
        param_grid2: dict[str, list] = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.001]}
        scoring2 = "f1_weighted"
        if optimize:
            gs2 = GridSearchCV(SVC(kernel="rbf", probability=True), param_grid2,
                               cv=min(5, len(y_train)), scoring=scoring2, n_jobs=-1)
            gs2.fit(X_train, y_train)
            print(f"  SVM best params: {gs2.best_params_}")
            return gs2.best_estimator_
        m2 = SVC(kernel="rbf", probability=True)
        m2.fit(X_train, y_train)
        return m2


# ---------------------------------------------------------------------------
# Evaluation & plotting
# ---------------------------------------------------------------------------

def _q2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _plot_regression(y_test, y_pred, model_name: str, output_dir: str, prefix: str):
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors="k")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--")
    ax.set_xlabel("Actual pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title(f"{model_name}\nR²={r2:.3f}  RMSE={rmse:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_predicted_vs_actual.png"), dpi=150)
    plt.close(fig)


def _plot_feature_importance(importance: np.ndarray, names: list[str],
                              model_name: str, output_dir: str, prefix: str, top_n: int = 20):
    top = min(top_n, len(names))
    idx = np.argsort(importance)[-top:]
    fig, ax = plt.subplots(figsize=(9, max(4, top * 0.35)))
    ax.barh(range(top), importance[idx])
    ax.set_yticks(range(top))
    ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top} Features – {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_feature_importance.png"), dpi=150)
    plt.close(fig)


def _plot_y_rand(result: dict, task: str, output_dir: str, prefix: str):
    metric = "R²" if task == "regression" else "F1"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(result["rand_scores"], bins=20, alpha=0.7, label=f"Randomized {metric}")
    ax.axvline(result["true_score"], color="red", linewidth=2, label=f"True model {metric}")
    ax.axvline(result["rand_mean"] + 2 * result["rand_std"], color="orange",
               linestyle="--", label="Mean + 2σ threshold")
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.set_title(f"Y-Randomization Test ({result['verdict']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_y_randomization.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class SmallDataQSAR:
    """
    End-to-end QSAR pipeline optimised for datasets with < 200 compounds.

    Parameters
    ----------
    task : 'regression' or 'classification'
    n_features_to_select : int, optional
        Target number of descriptors after RFE. Defaults to max(5, p//3).
    optimize_svm : bool
        Grid-search SVM hyperparameters (C, gamma). Default True.
    n_y_rand_trials : int
        Number of trials for the Y-randomization test. Default 100.
    """

    def __init__(
        self,
        task: str = "regression",
        n_features_to_select: int | None = None,
        optimize_svm: bool = True,
        n_y_rand_trials: int = 100,
    ):
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        self.task = task
        self.n_features_to_select = n_features_to_select
        self.optimize_svm = optimize_svm
        self.n_y_rand_trials = n_y_rand_trials

        # populated after fit_evaluate()
        self.scaler_: StandardScaler | None = None
        self.selector_: FeatureSelector | None = None
        self.label_encoder_: LabelEncoder | None = None
        self.models_: dict[str, Any] = {}
        self.metrics_: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def fit_evaluate(
        self,
        csv_path: str,
        target_col: str = "pIC50",
        test_size: float = 0.2,
        output_dir: str | None = None,
    ) -> dict:
        """
        Full pipeline: load → feature selection → train multiple models →
        LOOCV → Repeated K-Fold → Y-Randomization → save artefacts.

        Returns a dict of all metrics.
        """
        if output_dir is None:
            output_dir = f"models/{timestamp}/small_data_{self.task}"
        os.makedirs(output_dir, exist_ok=True)

        # ---- Load data ----
        X_raw, y_raw, feature_names, df = _load_xy(csv_path, target_col)
        print(f"\n{'='*70}")
        print(f"  SmallDataQSAR  task={self.task}  n={X_raw.shape[0]}  p={X_raw.shape[1]}")
        print(f"{'='*70}")

        # ---- Encode labels for classification ----
        if self.task == "classification":
            self.label_encoder_ = LabelEncoder()
            y: np.ndarray = np.asarray(self.label_encoder_.fit_transform(np.asarray(y_raw)))
            print(f"  Classes: {dict(zip(self.label_encoder_.classes_, range(len(self.label_encoder_.classes_))))}")
        else:
            y = np.asarray(y_raw, dtype=float)
            self.label_encoder_ = None

        # ---- Train / test split ----
        from sklearn.model_selection import train_test_split
        stratify_y = y if self.task == "classification" else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y, test_size=test_size, random_state=42, stratify=stratify_y
        )

        # ---- Feature selection (on training data only) ----
        print("\n[Step 1] Feature Selection")
        self.selector_ = FeatureSelector(
            n_features=self.n_features_to_select,
            task=self.task,
        )
        X_tr_sel = self.selector_.fit(X_tr, y_tr, feature_names)
        X_te_sel = self.selector_.transform(X_te)
        sel_names = self.selector_.selected_names_

        # Save selected feature names
        with open(os.path.join(output_dir, "selected_features.txt"), "w") as fh:
            fh.write("\n".join(sel_names))

        # ---- Scaling ----
        self.scaler_ = StandardScaler()
        X_tr_sc = self.scaler_.fit_transform(X_tr_sel)
        X_te_sc = self.scaler_.transform(X_te_sel)

        # Full dataset (scaled, selected) for CV
        X_full_sel = self.selector_.transform(X_raw)
        X_full_sc  = self.scaler_.transform(X_full_sel)

        # ---- Train models ----
        print("\n[Step 2] Model Training")
        model_configs = self._build_models(X_tr_sc, y_tr)

        all_results: dict[str, dict] = {}

        for name, model in model_configs.items():
            print(f"\n  --- {name} ---")
            prefix = name.lower().replace(" ", "_")

            # --- Hold-out evaluation ---
            y_pred = model.predict(X_te_sc)
            if self.task == "regression":
                r2 = float(r2_score(y_te, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                q2_ho = float(_q2_score(y_te, y_pred))
                print(f"  Hold-out  R²={r2:.3f}  RMSE={rmse:.3f}  Q²={q2_ho:.3f}")
                metrics: dict[str, Any] = {"r2": r2, "rmse": rmse, "q2_hold_out": q2_ho}
                _plot_regression(y_te, y_pred, name, output_dir, prefix)
            else:
                acc = float(accuracy_score(y_te, y_pred))
                f1  = float(f1_score(y_te, y_pred, average="weighted"))
                if len(np.unique(y_te)) == 2:
                    proba = model.predict_proba(X_te_sc)[:, 1]
                    auc = float(roc_auc_score(y_te, proba))
                    fpr, tpr, _ = roc_curve(y_te, proba)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                    ax.set_title(f"ROC – {name}"); ax.legend()
                    fig.tight_layout()
                    fig.savefig(os.path.join(output_dir, f"{prefix}_roc.png"), dpi=150)
                    plt.close(fig)
                else:
                    auc = float(roc_auc_score(y_te, model.predict_proba(X_te_sc),
                                              multi_class="ovr", average="weighted"))
                print(f"  Hold-out  Accuracy={acc:.3f}  F1={f1:.3f}  AUC-ROC={auc:.3f}")
                # Confusion matrix
                cm = confusion_matrix(y_te, y_pred)
                class_names = ([str(c) for c in self.label_encoder_.classes_]
                               if self.label_encoder_ else None)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=class_names or "auto",
                            yticklabels=class_names or "auto", ax=ax)
                ax.set_title(f"Confusion Matrix – {name}")
                ax.set_ylabel("True"); ax.set_xlabel("Predicted")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"), dpi=150)
                plt.close(fig)
                metrics = {"accuracy": acc, "f1": f1, "auc_roc": auc}

            # --- Feature importance ---
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                _plot_feature_importance(imp, sel_names, name, output_dir, prefix)
                pd.DataFrame({"feature": sel_names, "importance": imp}).sort_values(
                    "importance", ascending=False
                ).to_csv(os.path.join(output_dir, f"{prefix}_feature_importance.csv"), index=False)
                metrics["feature_importances"] = dict(zip(sel_names, imp.tolist()))

            # --- LOOCV ---
            print(f"  Running LOOCV…")
            loocv_res = loocv_score(model, X_full_sc, y, task=self.task)
            metrics["loocv"] = {k: v for k, v in loocv_res.items() if k != "y_cv"}

            # --- Repeated K-Fold ---
            print(f"  Running Repeated K-Fold…")
            n_splits = min(5, max(2, len(y) // 5))
            rkf_res = repeated_kfold_score(model, X_full_sc, y,
                                           n_splits=n_splits, task=self.task)
            metrics["repeated_kfold"] = rkf_res

            # --- Y-Randomization ---
            print(f"  Running Y-Randomization ({self.n_y_rand_trials} trials)…")
            yrand_res = y_randomization_test(model, X_full_sc, y,
                                             n_trials=self.n_y_rand_trials,
                                             task=self.task, cv=n_splits)
            metrics["y_randomization"] = yrand_res
            _plot_y_rand(yrand_res, self.task, output_dir, prefix)

            all_results[name] = metrics
            self.models_[name] = model

            # Save model
            with open(os.path.join(output_dir, f"{prefix}_model.pkl"), "wb") as fh:
                pickle.dump(model, fh)

        self.metrics_ = all_results

        # ---- Save scaler / selector ----
        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as fh:
            pickle.dump(self.scaler_, fh)
        with open(os.path.join(output_dir, "feature_selector.pkl"), "wb") as fh:
            pickle.dump(self.selector_, fh)
        if self.label_encoder_:
            with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as fh:
                pickle.dump(self.label_encoder_, fh)

        # ---- Comparison table ----
        self._print_summary(all_results, output_dir)

        return all_results

    # ------------------------------------------------------------------
    def _build_models(self, X_tr: np.ndarray, y_tr: np.ndarray) -> dict[str, Any]:
        """Train all baseline models for the task."""
        models: dict[str, Any] = {}

        if self.task == "regression":
            # PLS
            print("  Training PLS…")
            models["PLS"] = _build_pls(X_tr, y_tr)

            # SVM-RBF
            print("  Training SVM…")
            models["SVM"] = _build_svm(X_tr, y_tr, task="regression",
                                        optimize=self.optimize_svm)

            # Random Forest
            print("  Training Random Forest…")
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            models["Random Forest"] = rf

            # XGBoost (regularised for small data)
            print("  Training XGBoost…")
            xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=1.0, reg_lambda=5.0,
                               random_state=42, n_jobs=-1, eval_metric="rmse",
                               verbosity=0)
            xgb.fit(X_tr, y_tr)
            models["XGBoost"] = xgb

        else:  # classification
            # SVM-RBF
            print("  Training SVM…")
            models["SVM"] = _build_svm(X_tr, y_tr, task="classification",
                                        optimize=self.optimize_svm)

            # Random Forest
            print("  Training Random Forest…")
            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            models["Random Forest"] = rf

            # XGBoost (regularised)
            print("  Training XGBoost…")
            xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                reg_alpha=1.0, reg_lambda=5.0,
                                random_state=42, n_jobs=-1, eval_metric="logloss",
                                verbosity=0)
            xgb.fit(X_tr, y_tr)
            models["XGBoost"] = xgb

        return models

    # ------------------------------------------------------------------
    def _print_summary(self, results: dict, output_dir: str):
        print(f"\n{'='*70}")
        print("  Model Comparison Summary")
        print(f"{'='*70}")
        rows = []
        for model_name, m in results.items():
            row: dict[str, Any] = {"model": model_name}
            if self.task == "regression":
                row["R²"]          = round(m.get("r2", float("nan")), 3)
                row["RMSE"]        = round(m.get("rmse", float("nan")), 3)
                row["Q² (hold-out)"]  = round(m.get("q2_hold_out", float("nan")), 3)
                row["Q² (LOOCV)"]     = round(m.get("loocv", {}).get("q2", float("nan")), 3)
                row["R²-CV (mean)"]   = round(m.get("repeated_kfold", {}).get("mean", float("nan")), 3)
            else:
                row["Accuracy"]       = round(m.get("accuracy", float("nan")), 3)
                row["F1"]             = round(m.get("f1", float("nan")), 3)
                row["AUC-ROC"]        = round(m.get("auc_roc", float("nan")), 3)
                row["Acc-LOOCV"]      = round(m.get("loocv", {}).get("accuracy_cv", float("nan")), 3)
                row["F1-CV (mean)"]   = round(m.get("repeated_kfold", {}).get("mean", float("nan")), 3)
            row["Y-Rand passed"]   = m.get("y_randomization", {}).get("passed", None)
            rows.append(row)

        df_summary = pd.DataFrame(rows).set_index("model")
        print(df_summary.to_string())
        df_summary.to_csv(os.path.join(output_dir, "model_comparison.csv"))
        print(f"\n  Artefacts saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Small-Data QSAR pipeline")
    parser.add_argument("csv", help="Input CSV with descriptors/fingerprints")
    parser.add_argument("--target", default="pIC50", help="Target column name")
    parser.add_argument("--task", choices=["regression", "classification"],
                        default="regression")
    parser.add_argument("--n-features", type=int, default=None,
                        help="Number of features to select (RFE)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-svm-optimize", action="store_true")
    parser.add_argument("--y-rand-trials", type=int, default=100)
    args = parser.parse_args()

    qsar = SmallDataQSAR(
        task=args.task,
        n_features_to_select=args.n_features,
        optimize_svm=not args.no_svm_optimize,
        n_y_rand_trials=args.y_rand_trials,
    )
    results = qsar.fit_evaluate(args.csv, target_col=args.target,
                                 output_dir=args.output_dir)
    print("\nDone.")
