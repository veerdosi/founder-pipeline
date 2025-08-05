#!/usr/bin/env python3
"""
XGBoost Founder Success Prediction Model
========================================

This module implements the XGBoost-based founder success prediction model
with proper temporal validation, hyperparameter optimization, and comprehensive
evaluation metrics.

Features:
- Temporal train/validation/test split
- XGBoost with optimized hyperparameters
- Random Forest validation model
- Comprehensive evaluation metrics
- Feature importance analysis
- Model interpretability with SHAP
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, classification_report,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)

class FounderSuccessPredictor:
    """XGBoost-based founder success prediction model"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.xgb_model = None
        self.rf_model = None
        self.calibrated_xgb = None
        self.feature_columns = None
        self.feature_importance = None
        self.model_metrics = {}
        
        # Default XGBoost parameters (optimized for founder prediction)
        self.xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,      # L1 regularization
            'reg_lambda': 1.0,     # L2 regularization
            'scale_pos_weight': 2, # Handle class imbalance (adjust based on data)
            'random_state': random_state,
            'eval_metric': 'auc',
            'verbosity': 0
        }
        
        # Random Forest parameters for validation
        self.rf_params = {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': random_state,
            'n_jobs': -1
        }
    
    def temporal_split(self, df: pd.DataFrame, 
                      train_end_year: int = 2019,
                      val_start_year: int = 2020,
                      val_end_year: int = 2021,
                      test_start_year: int = 2022) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal split for time-series validation
        
        Args:
            df: DataFrame with founding_year column
            train_end_year: Last year for training data
            val_start_year: First year for validation data  
            val_end_year: Last year for validation data
            test_start_year: First year for test data
            
        Returns:
            train_df, val_df, test_df
        """
        logger.info("Creating temporal split...")
        
        # Training set: 2013-2019 (7 years)
        train_mask = df['founding_year'] <= train_end_year
        train_df = df[train_mask].copy()
        
        # Validation set: 2020-2021 (2 years)
        val_mask = (df['founding_year'] >= val_start_year) & (df['founding_year'] <= val_end_year)
        val_df = df[val_mask].copy()
        
        # Test set: 2022+ (recent data)
        test_mask = df['founding_year'] >= test_start_year
        test_df = df[test_mask].copy()
        
        logger.info(f"Train set: {len(train_df)} companies ({train_df['founding_year'].min()}-{train_df['founding_year'].max()})")
        logger.info(f"Val set: {len(val_df)} companies ({val_df['founding_year'].min()}-{val_df['founding_year'].max() if len(val_df) > 0 else 'N/A'})")
        logger.info(f"Test set: {len(test_df)} companies ({test_df['founding_year'].min()}-{test_df['founding_year'].max() if len(test_df) > 0 else 'N/A'})")
        
        return train_df, val_df, test_df
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Optimize XGBoost hyperparameters using GridSearchCV"""
        logger.info("Optimizing hyperparameters...")
        
        # Reduced parameter grid for faster optimization
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [1.0, 2.0]
        }
        
        # Use TimeSeriesSplit for cross-validation within training data
        cv = TimeSeriesSplit(n_splits=5)
        
        # Create base model
        base_model = xgb.XGBClassifier(**{k: v for k, v in self.xgb_params.items() 
                                         if k not in param_grid})
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update parameters
        self.xgb_params.update(grid_search.best_params_)
        
        return grid_search.best_params_
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None,
                    optimize_hyperparams: bool = True) -> Dict:
        """Train XGBoost and Random Forest models"""
        logger.info("Training models...")
        
        self.feature_columns = X_train.columns.tolist()
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams and X_val is not None:
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Adjust scale_pos_weight based on actual class distribution
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        self.xgb_params['scale_pos_weight'] = pos_weight
        logger.info(f"Adjusted scale_pos_weight to: {pos_weight:.2f}")
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        
        # Use validation set for early stopping if available
        if X_val is not None and y_val is not None:
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train, y_train)
        
        # Train Random Forest for validation
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # Calibrate XGBoost probabilities
        self.calibrated_xgb = CalibratedClassifierCV(self.xgb_model, method='isotonic', cv=3)
        self.calibrated_xgb.fit(X_train, y_train)
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance_xgb': self.xgb_model.feature_importances_,
            'importance_rf': self.rf_model.feature_importances_
        }).sort_values('importance_xgb', ascending=False)
        
        logger.info("Model training completed successfully")
        
        return {
            'xgb_params': self.xgb_params,
            'rf_params': self.rf_params,
            'feature_count': len(self.feature_columns)
        }
    
    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained models
        
        Args:
            X: Feature matrix
            use_ensemble: If True, combine XGBoost and Random Forest predictions
            
        Returns:
            predictions, probabilities
        """
        if self.xgb_model is None:
            raise ValueError("Models must be trained first")
        
        # Ensure feature columns match
        X_pred = X[self.feature_columns]
        
        # Get predictions from both models
        xgb_proba = self.calibrated_xgb.predict_proba(X_pred)[:, 1]
        
        if use_ensemble:
            rf_proba = self.rf_model.predict_proba(X_pred)[:, 1]
            # Ensemble: 70% XGBoost, 30% Random Forest
            ensemble_proba = 0.7 * xgb_proba + 0.3 * rf_proba
            predictions = (ensemble_proba >= 0.5).astype(int)
            return predictions, ensemble_proba
        else:
            predictions = (xgb_proba >= 0.5).astype(int)
            return predictions, xgb_proba
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, 
                      dataset_name: str = "test") -> Dict:
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating model on {dataset_name} set...")
        
        # Get predictions
        predictions, probabilities = self.predict(X_test, use_ensemble=True)
        
        # Calculate metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'roc_auc': roc_auc_score(y_test, probabilities),
        }
        
        # Precision at 65% threshold (business requirement)
        high_conf_predictions = (probabilities >= 0.65).astype(int)
        if high_conf_predictions.sum() > 0:
            metrics['precision_at_65%'] = precision_score(y_test, high_conf_predictions)
            metrics['recall_at_65%'] = recall_score(y_test, high_conf_predictions)
            metrics['companies_flagged_at_65%'] = high_conf_predictions.sum()
        else:
            metrics['precision_at_65%'] = 0.0
            metrics['recall_at_65%'] = 0.0
            metrics['companies_flagged_at_65%'] = 0
        
        # Store metrics
        self.model_metrics[dataset_name] = metrics
        
        logger.info(f"{dataset_name.capitalize()} Metrics:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision@65%: {metrics['precision_at_65%']:.4f}")
        
        return metrics
    
    def plot_evaluation_charts(self, X_test: pd.DataFrame, y_test: pd.Series, 
                             save_path: str = None):
        """Create comprehensive evaluation charts"""
        
        predictions, probabilities = self.predict(X_test, use_ensemble=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Founder Success Prediction Model Evaluation', fontsize=16)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_score = roc_auc_score(y_test, probabilities)
        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        axes[0, 1].plot(recall, precision)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature Importance (Top 15)
        top_features = self.feature_importance.head(15)
        axes[0, 2].barh(range(len(top_features)), top_features['importance_xgb'])
        axes[0, 2].set_yticks(range(len(top_features)))
        axes[0, 2].set_yticklabels(top_features['feature'])
        axes[0, 2].set_xlabel('Importance')
        axes[0, 2].set_title('Top 15 Feature Importance (XGBoost)')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # Probability Distribution
        axes[1, 1].hist(probabilities[y_test == 0], bins=30, alpha=0.7, label='Not Successful', density=True)
        axes[1, 1].hist(probabilities[y_test == 1], bins=30, alpha=0.7, label='Successful', density=True)
        axes[1, 1].axvline(x=0.65, color='red', linestyle='--', label='65% Threshold')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Probability Distribution')
        axes[1, 1].legend()
        
        # Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, probabilities, n_bins=10
        )
        axes[1, 2].plot(mean_predicted_value, fraction_of_positives, "s-", label='Model')
        axes[1, 2].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[1, 2].set_xlabel('Mean Predicted Probability')
        axes[1, 2].set_ylabel('Fraction of Positives')
        axes[1, 2].set_title('Calibration Plot')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation charts saved to {save_path}")
        
        plt.show()
    
    def analyze_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Analyze and display feature importance"""
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        top_features = self.feature_importance.head(top_n)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("=" * 60)
        print(f"{'Rank':<4} {'Feature':<30} {'XGB Imp':<10} {'RF Imp':<10}")
        print("-" * 60)
        
        for i, row in top_features.iterrows():
            print(f"{i+1:<4} {row['feature'][:30]:<30} {row['importance_xgb']:<10.4f} {row['importance_rf']:<10.4f}")
        
        return top_features
    
    def generate_shap_explanations(self, X_sample: pd.DataFrame, max_samples: int = 100):
        """Generate SHAP explanations for model interpretability"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Cannot generate explanations.")
            return None
        
        if self.xgb_model is None:
            raise ValueError("Model must be trained first")
        
        # Sample data for SHAP (can be expensive for large datasets)
        X_shap = X_sample.sample(min(max_samples, len(X_sample)))[self.feature_columns]
        
        # Create SHAP explainer
        explainer = shap.Explainer(self.xgb_model)
        shap_values = explainer(X_shap)
        
        # Summary plot
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.title('SHAP Feature Importance Summary')
        plt.tight_layout()
        plt.show()
        
        return shap_values
    
    def save_model(self, filepath: str):
        """Save trained model and metadata"""
        model_data = {
            'xgb_model': self.xgb_model,
            'rf_model': self.rf_model,
            'calibrated_xgb': self.calibrated_xgb,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'xgb_params': self.xgb_params,
            'rf_params': self.rf_params,
            'model_metrics': self.model_metrics,
            'training_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and metadata"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.xgb_model = model_data['xgb_model']
        self.rf_model = model_data['rf_model']
        self.calibrated_xgb = model_data['calibrated_xgb']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data['feature_importance']
        self.xgb_params = model_data['xgb_params']
        self.rf_params = model_data['rf_params']
        self.model_metrics = model_data['model_metrics']
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """Main training and evaluation pipeline"""
    
    # Load processed dataset
    logger.info("Loading ML-ready dataset...")
    df = pd.read_csv("/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/ml_ready_dataset.csv")
    
    # Initialize predictor
    predictor = FounderSuccessPredictor()
    
    # Create temporal split
    train_df, val_df, test_df = predictor.temporal_split(df)
    
    # Prepare features
    from feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    
    # Training data
    X_train, y_train = fe.prepare_features_for_ml(train_df, target_col='is_successful')
    
    # Validation data (if available)
    if len(val_df) > 0:
        X_val, y_val = fe.prepare_features_for_ml(val_df, target_col='is_successful')
    else:
        X_val, y_val = None, None
    
    # Test data (if available)
    if len(test_df) > 0:
        X_test, y_test = fe.prepare_features_for_ml(test_df, target_col='is_successful')
    else:
        X_test, y_test = None, None
    
    # Train models
    training_info = predictor.train_models(X_train, y_train, X_val, y_val, optimize_hyperparams=False)
    
    # Evaluate on all available datasets
    predictor.evaluate_model(X_train, y_train, "train")
    
    if X_val is not None:
        predictor.evaluate_model(X_val, y_val, "validation")
    
    if X_test is not None:
        predictor.evaluate_model(X_test, y_test, "test")
        
        # Create evaluation charts
        predictor.plot_evaluation_charts(
            X_test, y_test, 
            save_path="/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/model_evaluation.png"
        )
    
    # Analyze feature importance
    predictor.analyze_feature_importance(top_n=20)
    
    # Generate SHAP explanations (if available)
    if len(test_df) > 0:
        predictor.generate_shap_explanations(X_test, max_samples=50)
    
    # Save model
    model_path = "/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/founder_success_model.pkl"
    predictor.save_model(model_path)
    
    # Save metrics to JSON
    metrics_path = "/Users/veerdosi/Documents/code/github/initiation-pipeline/datasets/model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(predictor.model_metrics, f, indent=2)
    
    logger.info("Training pipeline completed successfully!")
    
    return predictor


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    predictor = main()