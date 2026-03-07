"""
predictive_extensions.py
========================
Part 2 of new_development.txt – Predictive Extensions for QSAR pipelines.

Modules
-------
1. ApplicabilityDomain   – Tanimoto- or leverage-based AD filtering
2. ConformalPredictor    – Model-agnostic prediction intervals / sets
3. AdmetFilter           – SwissADME / pkCSM API hooks + multi-objective scoring
4. MultiTaskQSAR         – Multi-output classifier/regressor for related targets

Quick start
-----------
    from predictive_extensions import ApplicabilityDomain, ConformalPredictor, AdmetFilter

    # --- Applicability Domain ---
    ad = ApplicabilityDomain(method='tanimoto', threshold=0.4)
    ad.fit(X_train_fps)          # fingerprint bit-vectors
    mask = ad.predict(X_query)   # True = inside AD

    # --- Conformal prediction (regression) ---
    cp = ConformalPredictor(base_model=rf_model, task='regression', alpha=0.1)
    cp.calibrate(X_cal, y_cal)
    intervals = cp.predict(X_new)    # DataFrame with lo / hi / inside_ad

    # --- ADMET ---
    admet = AdmetFilter()
    df_admet = admet.query_swissadme(smiles_list)
    ranked   = admet.rank_compounds(results_df, qsar_col='Prob_1',
                                     admet_cols=['HIA', 'BBB'])

    # --- Multi-task ---
    mt = MultiTaskQSAR(task='classification')
    mt.fit(X_train, Y_train)         # Y_train: (n_samples, n_targets)
    Y_pred = mt.predict(X_test)
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from sklearn.base import clone
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# 1. Applicability Domain
# ===========================================================================

class ApplicabilityDomain:
    """
    Applicability Domain (AD) estimation.

    Two methods are supported:

    ``tanimoto`` (default for fingerprints)
        A query compound is inside the AD if its *k*-nearest-neighbour
        average Tanimoto similarity to the training set exceeds ``threshold``.

    ``leverage`` (for real-valued descriptors / scaled features)
        Uses the hat-matrix (leverage) approach.  A compound with leverage
        h > h* = 3(p+1)/n is flagged as outside the AD (Williams plot).
    """

    def __init__(
        self,
        method: str = "tanimoto",
        threshold: float = 0.4,
        k: int = 5,
    ):
        if method not in ("tanimoto", "leverage"):
            raise ValueError("method must be 'tanimoto' or 'leverage'")
        self.method = method
        self.threshold = threshold
        self.k = k

        self._X_train: np.ndarray | None = None
        self._hat_threshold: float | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray) -> "ApplicabilityDomain":
        """Memorise training data (tanimoto) or compute hat-matrix (leverage)."""
        self._X_train = X_train.copy()
        n, p = X_train.shape

        if self.method == "leverage":
            # hat threshold h* = 3(p+1)/n
            self._hat_threshold = 3.0 * (p + 1) / n
            # pseudo-inverse for leverage calculation
            try:
                self._Xt_Xt_inv = np.linalg.pinv(X_train.T @ X_train)
            except np.linalg.LinAlgError:
                self._Xt_Xt_inv = np.eye(p)

        print(f"  AD fitted on {n} training compounds (method={self.method}, "
              f"threshold={self.threshold})")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask: True = inside AD, False = outside."""
        assert self._X_train is not None, "Call fit() first."

        if self.method == "tanimoto":
            return self._tanimoto_ad(X)
        else:
            return self._leverage_ad(X)

    # ------------------------------------------------------------------
    def _tanimoto_ad(self, X: np.ndarray) -> np.ndarray:
        """Bit-vector Tanimoto similarity against training set."""
        X_tr = self._X_train.astype(bool)  # type: ignore[union-attr]
        X_q  = X.astype(bool)
        inside = np.zeros(len(X_q), dtype=bool)
        for i, q in enumerate(X_q):
            # Jaccard / Tanimoto for bit vectors
            inter = (X_tr & q).sum(axis=1)
            union = (X_tr | q).sum(axis=1)
            sim = np.where(union > 0, inter / union, 0.0)
            top_k = np.sort(sim)[-self.k:]
            inside[i] = top_k.mean() >= self.threshold
        return inside

    # ------------------------------------------------------------------
    def _leverage_ad(self, X: np.ndarray) -> np.ndarray:
        """Hat-matrix leverage for scaled descriptor vectors."""
        h_vals = np.array([
            float(x @ self._Xt_Xt_inv @ x)  # type: ignore[operator]
            for x in X
        ])
        return h_vals <= self._hat_threshold  # type: ignore[operator]

    # ------------------------------------------------------------------
    def williams_plot(
        self,
        X_train: np.ndarray,
        residuals_train: np.ndarray,
        X_test: np.ndarray,
        residuals_test: np.ndarray,
        output_path: str = "williams_plot.png",
    ):
        """Generate a Williams plot (leverage vs. standardised residuals)."""
        assert self.method == "leverage", "Williams plot requires method='leverage'"

        def leverage(X: np.ndarray) -> np.ndarray:
            return np.array([float(x @ self._Xt_Xt_inv @ x)  # type: ignore[operator]
                             for x in X])

        h_tr = leverage(X_train)
        h_te = leverage(X_test)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(h_tr, residuals_train, c="steelblue", alpha=0.7, label="Train")
        ax.scatter(h_te, residuals_test,  c="tomato",    alpha=0.7, label="Test")
        assert self._hat_threshold is not None
        ax.axvline(self._hat_threshold, color="k", linestyle="--",
                   label=f"h* = {self._hat_threshold:.3f}")
        ax.axhline( 3, color="orange", linestyle="--", label="±3σ")
        ax.axhline(-3, color="orange", linestyle="--")
        ax.set_xlabel("Leverage (h)")
        ax.set_ylabel("Standardised residual")
        ax.set_title("Williams Plot (Applicability Domain)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"  Williams plot saved to {output_path}")


# ===========================================================================
# 2. Conformal Predictor
# ===========================================================================

class ConformalPredictor:
    """
    Inductive Conformal Prediction (ICP) wrapper around any sklearn-compatible
    fitted base model.

    Regression
    ----------
    Nonconformity score: |y_true – y_pred|
    Prediction intervals are guaranteed to contain the true value with
    probability ≥ (1 – alpha) under exchangeability.

    Classification
    --------------
    Nonconformity score: 1 – P(true class)
    Prediction *sets* are returned: the set of classes whose
    nonconformity score ≤ the alpha-quantile of the calibration set.

    Parameters
    ----------
    base_model : fitted sklearn estimator
    task : 'regression' or 'classification'
    alpha : float
        Significance level (default 0.1 → 90% coverage).
    """

    def __init__(self, base_model: Any, task: str = "regression", alpha: float = 0.1):
        self.base_model = base_model
        self.task = task
        self.alpha = alpha
        self._q_hat: float | None = None          # regression quantile
        self._cal_scores: np.ndarray | None = None  # classification NC scores

    # ------------------------------------------------------------------
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Compute nonconformity scores on a held-out calibration set."""
        if self.task == "regression":
            y_pred = self.base_model.predict(X_cal).ravel()
            scores = np.abs(y_cal.ravel() - y_pred)
            n = len(scores)
            # Conformal quantile with finite-sample correction
            level = np.ceil((n + 1) * (1 - self.alpha)) / n
            level = min(level, 1.0)
            self._q_hat = float(np.quantile(scores, level))
            print(f"  Conformal regression calibrated  q̂ = {self._q_hat:.4f} "
                  f"(α={self.alpha}, n_cal={n})")
        else:
            proba = self.base_model.predict_proba(X_cal)
            y_enc = y_cal.astype(int)
            self._cal_scores = 1.0 - proba[np.arange(len(y_enc)), y_enc]
            assert self._cal_scores is not None
            n = len(self._cal_scores)
            level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self._class_threshold = float(np.quantile(self._cal_scores, min(level, 1.0)))
            print(f"  Conformal classification calibrated  τ = {self._class_threshold:.4f} "
                  f"(α={self.alpha}, n_cal={n})")
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        ad: ApplicabilityDomain | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with predictions and uncertainty information.

        Regression columns : y_pred, lo, hi, interval_width, [inside_ad]
        Classification columns : predicted_class, prediction_set, [inside_ad]
        """
        assert self._q_hat is not None or self._cal_scores is not None, \
            "Call calibrate() first."

        if self.task == "regression":
            y_pred = self.base_model.predict(X).ravel()
            lo = y_pred - self._q_hat  # type: ignore[operator]
            hi = y_pred + self._q_hat  # type: ignore[operator]
            df = pd.DataFrame({
                "y_pred": y_pred,
                "lo":     lo,
                "hi":     hi,
                "interval_width": hi - lo,
            })
        else:
            proba = self.base_model.predict_proba(X)
            nc_scores = 1.0 - proba          # shape (n, n_classes)
            pred_sets = [
                list(np.where(nc_scores[i] <= self._class_threshold)[0])  # type: ignore[operator]
                for i in range(len(X))
            ]
            pred_class = self.base_model.predict(X)
            df = pd.DataFrame({
                "predicted_class": pred_class,
                "prediction_set":  pred_sets,
                "set_size":        [len(s) for s in pred_sets],
            })

        if ad is not None:
            df["inside_ad"] = ad.predict(X)

        return df

    # ------------------------------------------------------------------
    def coverage_report(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Empirical coverage on a test set (should be ≥ 1-alpha).
        """
        df = self.predict(X_test)
        if self.task == "regression":
            covered = ((y_test >= df["lo"]) & (y_test <= df["hi"])).mean()
            mean_width = df["interval_width"].mean()
            print(f"  Empirical coverage: {covered:.3f} (target ≥ {1-self.alpha:.2f})")
            print(f"  Mean interval width: {mean_width:.4f}")
            return {"coverage": float(covered), "mean_width": float(mean_width)}
        else:
            pred_sets = df["prediction_set"].tolist()
            covered = np.mean([int(y_test[i]) in pred_sets[i] for i in range(len(y_test))])
            mean_size = df["set_size"].mean()
            print(f"  Empirical coverage: {covered:.3f} (target ≥ {1-self.alpha:.2f})")
            print(f"  Mean prediction set size: {mean_size:.2f}")
            return {"coverage": float(covered), "mean_set_size": float(mean_size)}


# ===========================================================================
# 3. ADMET Filter
# ===========================================================================

_SWISSADME_URL = "http://www.swissadme.ch/include/smiles2json.php"
_PKCSM_URL     = "https://biosig.lab.uq.edu.au/pkcsm/api/pkcsm_prediction"

_PKCSM_ENDPOINTS = [
    "absorption-water-solubility",
    "absorption-caco2-permeability",
    "absorption-intestinal-absorption",
    "absorption-skin-permeability",
    "distribution-vd",
    "distribution-bbb-permeability",
    "distribution-cns-permeability",
    "metabolism-cyp1a2-inhibitor",
    "metabolism-cyp2c19-inhibitor",
    "metabolism-cyp2c9-inhibitor",
    "metabolism-cyp2d6-inhibitor",
    "metabolism-cyp3a4-inhibitor",
    "toxicity-ames",
    "toxicity-herg",
]


class AdmetFilter:
    """
    ADMET property hooks via SwissADME and pkCSM public APIs.

    The APIs are queried over HTTP; an internet connection is required.
    Results are cached per SMILES string to avoid redundant requests.

    Methods
    -------
    query_swissadme(smiles_list)  →  DataFrame
    query_pkcsm(smiles_list)      →  DataFrame
    rank_compounds(df, qsar_col, admet_cols, weights)  →  DataFrame
    """

    def __init__(self, request_delay: float = 1.0):
        """
        Parameters
        ----------
        request_delay : float
            Seconds to wait between API calls (be polite to public servers).
        """
        self.request_delay = request_delay
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def query_swissadme(
        self, smiles_list: list[str], timeout: int = 30
    ) -> pd.DataFrame:
        """
        Query SwissADME for physicochemical and ADMET properties.

        Key returned columns (subset):
          MW, LogP, HBA, HBD, TPSA, RotBonds,
          GI_absorption, BBB_permeant, Pgp_substrate,
          CYP1A2_inhibitor, CYP2C9_inhibitor, CYP2D6_inhibitor, CYP3A4_inhibitor,
          Lipinski, Bioavailability_Score
        """
        records = []
        for smi in smiles_list:
            if smi in self._cache:
                records.append({"SMILES": smi, **self._cache[smi]})
                continue
            try:
                resp = requests.post(
                    _SWISSADME_URL,
                    data={"smiles": smi},
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                # SwissADME returns a list; take first entry
                props = data[0] if isinstance(data, list) else data
                flat = self._flatten_swissadme(props)
                self._cache[smi] = flat
                records.append({"SMILES": smi, **flat})
            except Exception as exc:
                print(f"  [SwissADME] Error for {smi[:30]}…: {exc}")
                records.append({"SMILES": smi})
            time.sleep(self.request_delay)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_swissadme(d: dict) -> dict:
        """Flatten nested SwissADME JSON."""
        flat: dict = {}
        for section, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[k2] = v2
            else:
                flat[section] = v
        return flat

    # ------------------------------------------------------------------
    def query_pkcsm(
        self, smiles_list: list[str], timeout: int = 60
    ) -> pd.DataFrame:
        """
        Query pkCSM for ADMET predictions.

        Returns a wide DataFrame with one column per property per SMILES.
        """
        records = []
        for smi in smiles_list:
            row: dict[str, Any] = {"SMILES": smi}
            for endpoint in _PKCSM_ENDPOINTS:
                cache_key = f"pkcsm::{endpoint}::{smi}"
                if cache_key in self._cache:
                    row.update(self._cache[cache_key])
                    continue
                try:
                    resp = requests.post(
                        f"{_PKCSM_URL}/{endpoint}",
                        json={"smiles": smi},
                        timeout=timeout,
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    prop = endpoint.replace("-", "_")
                    value = result.get("prediction", result.get("value", None))
                    row[prop] = value
                    self._cache[cache_key] = {prop: value}
                except Exception as exc:
                    print(f"  [pkCSM:{endpoint}] Error for {smi[:30]}…: {exc}")
                time.sleep(self.request_delay)
            records.append(row)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    def rank_compounds(
        self,
        df: pd.DataFrame,
        qsar_col: str,
        admet_cols: list[str] | None = None,
        weights: dict[str, float] | None = None,
        higher_is_better: dict[str, bool] | None = None,
    ) -> pd.DataFrame:
        """
        Multi-objective scoring: combine QSAR prediction with ADMET liabilities
        into a single composite score.

        Parameters
        ----------
        df : DataFrame
            Must contain ``qsar_col`` and any ``admet_cols``.
        qsar_col : str
            Column holding the QSAR activity score (higher = more active).
        admet_cols : list of str, optional
            ADMET columns to include in the composite score.
        weights : dict {col: float}, optional
            Weight for each column. Defaults to equal weights.
        higher_is_better : dict {col: bool}, optional
            True (default) = higher value is better; False = lower is better
            (e.g. toxicity columns).

        Returns
        -------
        DataFrame sorted by composite_score descending.
        """
        df = df.copy()
        cols = [qsar_col] + (admet_cols or [])
        cols = [c for c in cols if c in df.columns]

        if weights is None:
            weights = {c: 1.0 for c in cols}
        if higher_is_better is None:
            higher_is_better = {c: True for c in cols}

        # Min-max scale each column then weight
        composite = np.zeros(len(df))
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(float)
            lo, hi = vals.min(), vals.max()
            scaled = (vals - lo) / (hi - lo + 1e-9)
            if not higher_is_better.get(col, True):
                scaled = 1.0 - scaled
            composite += weights.get(col, 1.0) * scaled

        df["composite_score"] = composite
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        return df


# ===========================================================================
# 4. Multi-Task QSAR
# ===========================================================================

class MultiTaskQSAR:
    """
    Multi-output QSAR model for simultaneous prediction against several
    related targets.

    This compensates for small per-target training sets by leveraging
    shared structural information across targets.

    Parameters
    ----------
    task : 'regression' or 'classification'
    base_model : str
        'rf' (Random Forest, default), 'xgb' (XGBoost), or 'svm'.
    """

    def __init__(self, task: str = "regression", base_model: str = "rf"):
        self.task = task
        self.base_model_name = base_model
        self.scaler_: StandardScaler | None = None
        self.model_: MultiOutputRegressor | MultiOutputClassifier | None = None
        self.target_names_: list[str] = []

    # ------------------------------------------------------------------
    def _make_base(self):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from xgboost import XGBRegressor, XGBClassifier

        if self.base_model_name == "xgb":
            if self.task == "regression":
                return XGBRegressor(n_estimators=200, max_depth=3, random_state=42,
                                    verbosity=0, n_jobs=-1)
            else:
                return XGBClassifier(n_estimators=200, max_depth=3, random_state=42,
                                     verbosity=0, n_jobs=-1, eval_metric="logloss")
        elif self.base_model_name == "svm":
            from sklearn.svm import SVR, SVC
            return SVR(kernel="rbf") if self.task == "regression" else SVC(kernel="rbf", probability=True)
        else:  # rf
            if self.task == "regression":
                return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            else:
                return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                              random_state=42, n_jobs=-1)

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray | pd.DataFrame,
        target_names: list[str] | None = None,
    ) -> "MultiTaskQSAR":
        """
        Parameters
        ----------
        X : (n_samples, n_features)
        Y : (n_samples, n_targets)
        target_names : labels for each target column
        """
        if isinstance(Y, pd.DataFrame):
            self.target_names_ = Y.columns.tolist()
            Y = Y.values
        else:
            self.target_names_ = target_names or [f"target_{i}" for i in range(Y.shape[1])]

        self.scaler_ = StandardScaler()
        X_sc = self.scaler_.fit_transform(X)

        base = self._make_base()
        if self.task == "regression":
            self.model_ = MultiOutputRegressor(base, n_jobs=-1)
        else:
            self.model_ = MultiOutputClassifier(base, n_jobs=-1)

        self.model_.fit(X_sc, Y)
        print(f"  MultiTaskQSAR trained on {X.shape[0]} samples, "
              f"{Y.shape[1]} targets: {self.target_names_}")
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Return predictions as a DataFrame with one column per target."""
        assert self.model_ is not None, "Call fit() first."
        assert self.scaler_ is not None
        X_sc = self.scaler_.transform(X)
        Y_pred = self.model_.predict(X_sc)
        return pd.DataFrame(np.asarray(Y_pred), columns=self.target_names_)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        (Classification only) Return per-target probability arrays.
        """
        assert self.task == "classification", "predict_proba only for classification."
        assert self.model_ is not None
        assert self.scaler_ is not None
        X_sc = self.scaler_.transform(X)
        probas: dict[str, np.ndarray] = {}
        for i, est in enumerate(self.model_.estimators_):  # type: ignore[union-attr]
            probas[self.target_names_[i]] = est.predict_proba(X_sc)
        return probas

    # ------------------------------------------------------------------
    def cross_validate(
        self, X: np.ndarray, Y: np.ndarray | pd.DataFrame,
        n_splits: int = 5, n_repeats: int = 3,
    ) -> pd.DataFrame:
        """
        Repeated K-Fold CV per target; returns a per-target metrics DataFrame.
        """
        from sklearn.model_selection import KFold, cross_val_score

        if isinstance(Y, pd.DataFrame):
            Y_arr = Y.values
        else:
            Y_arr = Y

        assert self.scaler_ is not None
        X_sc = self.scaler_.transform(X) if self.model_ is not None else X

        scoring = "r2" if self.task == "regression" else "f1_weighted"
        rows = []
        for i, name in enumerate(self.target_names_):
            y_i = Y_arr[:, i]
            base = clone(self._make_base())
            scores_all = []
            for seed in range(n_repeats):
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                s  = cross_val_score(base, X_sc, y_i, cv=kf, scoring=scoring, n_jobs=-1)
                scores_all.extend(s.tolist())
            rows.append({
                "target": name,
                f"mean_{scoring}": float(np.mean(scores_all)),
                f"std_{scoring}":  float(np.std(scores_all)),
            })
        df = pd.DataFrame(rows).set_index("target")
        print(df.to_string())
        return df

    # ------------------------------------------------------------------
    def save(self, output_dir: str):
        """Persist the fitted model to disk."""
        import pickle
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "multitask_model.pkl"), "wb") as fh:
            pickle.dump(self, fh)
        print(f"  MultiTaskQSAR saved to {output_dir}/multitask_model.pkl")

    @classmethod
    def load(cls, output_dir: str) -> "MultiTaskQSAR":
        import pickle
        path = os.path.join(output_dir, "multitask_model.pkl")
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        print(f"  MultiTaskQSAR loaded from {path}")
        return obj


# ===========================================================================
# Demo / smoke-test
# ===========================================================================

if __name__ == "__main__":
    import argparse
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Predictive extensions demo")
    parser.add_argument("csv", help="Input CSV")
    parser.add_argument("--target", default="pIC50")
    parser.add_argument("--task", choices=["regression", "classification"],
                        default="regression")
    parser.add_argument("--smiles-col", default="canonical_smiles",
                        help="Column containing SMILES (for ADMET and AD)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    target_col = args.target
    exclude = ["molecule_chembl_id", "canonical_smiles", "Smiles",
               "pIC50", "activity_class", "Activity_Level"]
    feat_cols = [c for c in df.columns if c not in exclude and c != target_col]

    X = df[feat_cols].fillna(0).values.astype(float)
    y = df[target_col].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    X_cal, X_te2, y_cal, y_te2 = train_test_split(X_te, y_te, test_size=0.5, random_state=0)

    # --- Base model ---
    if args.task == "regression":
        base = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_tr2, _, y_tr2, _ = train_test_split(X_sc, y_enc, test_size=0.2, random_state=42)
        base = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr2, y_tr2)  # type: ignore[assignment]

    # --- Applicability Domain ---
    print("\n[1] Applicability Domain (leverage)")
    ad = ApplicabilityDomain(method="leverage")
    ad.fit(X_tr)
    inside = ad.predict(X_te)
    print(f"  {inside.sum()} / {len(inside)} test compounds inside AD")

    # --- Conformal Predictor ---
    print("\n[2] Conformal Predictor")
    cp = ConformalPredictor(base, task=args.task, alpha=0.1)
    cp.calibrate(X_cal, y_cal if args.task == "regression" else y_enc[:len(y_cal)])  # type: ignore[possibly-undefined]
    preds = cp.predict(X_te2, ad=ad)
    print(preds.head())
    cp.coverage_report(X_te2, y_te2 if args.task == "regression" else y_enc[:len(y_te2)])  # type: ignore[possibly-undefined]

    print("\nDone.")
