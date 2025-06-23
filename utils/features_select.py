# -*- encoding: utf-8 -*-
'''
@File    :   features_select.py
@Time    :   2025/04/19 21:11:51
@Author  :   myz 
'''
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, VarianceThreshold, RFE, SelectFromModel


def adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_selection_func):
    """
    Adaptive selection of topk_ratio by testing multiple values and selecting the one with the best validation performance.
    
    Parameters:
    X_train, y_train, X_val, y_val: Training and validation data.
    feature_selection_func: Function handle for feature selection method.
    
    Returns:
    best_topk_ratio (float): Best performing topk_ratio.
    """
    candidate_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_mse = float('inf')
    best_topk_ratio = None

    for ratio in candidate_ratios:
        selected_features = feature_selection_func(X_train, y_train, X_val, y_val, topk_ratio=ratio)
        if len(selected_features) == 0:
            continue  # Skip if no features are selected
        
        # Train a Random Forest model on selected features
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train[selected_features], y_train)
        # Evaluate on validation set
        y_pred = model.predict(X_val[selected_features])
        mse = mean_squared_error(y_val, y_pred)
        
        logger.info(f"topk_ratio={ratio}, MSE={mse}")
        if mse < best_mse:
            best_mse = mse
            best_topk_ratio = ratio
    
    logger.info(f"Best topk_ratio={best_topk_ratio} with MSE={best_mse}")
    return best_topk_ratio


def features_select(X_train, y_train, X_val, y_val, topk_ratio=None):
    """
    Feature selection combining XGBoost, LightGBM, and CatBoost feature importance with SHAP values.
    """
    from lightgbm import early_stopping, log_evaluation
    X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
    X_val, y_val = pd.DataFrame(X_val), pd.DataFrame(y_val)

    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, features_select)
    
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))  #

    def get_top_features_from_model(model, X_train_subset, topk):
        """
        Extract Feature Importance from the Model and Return the Top k Features
        """
        if hasattr(model, 'feature_importances_'):  # LightGBM 和 CatBoost
            importance = model.feature_importances_
        elif hasattr(model, 'get_score'):  # XGBoost
            importance_dict = model.get_score(importance_type='gain')
            importance = np.zeros(X_train_subset.shape[1])
            for i, col in enumerate(X_train_subset.columns):
                importance[i] = importance_dict.get(col, 0)
        else:
            raise ValueError("Unsupported model type.")
        
        feature_importance = pd.DataFrame({
            'feature': X_train_subset.columns,
            'importance': importance
        }).sort_values(by='importance', ascending=False)
        return feature_importance['feature'].iloc[:topk].tolist()

    def get_shap_top_features(model, X_train_subset, topk):
        """
        Calculate Feature Importance Using SHAP Values and Return the Top k Features
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_subset)
        shap_sum = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X_train_subset.columns,
            'importance': shap_sum
        }).sort_values(by='importance', ascending=False)
        return feature_importance['feature'].iloc[:topk].tolist()

    # model init 
    xgb_model = xgb.XGBRegressor()
    lgb_model = lgb.LGBMRegressor()
    cat_model = CatBoostRegressor(silent=True)

    # Train the Model and Extract Feature Importance
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=10, verbose=False),  # early stopping
            log_evaluation(period=0)  # 
        ]
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=10,
        verbose=False
    )

    # Extract Feature Importance for Each Model
    xgb_top_features = get_top_features_from_model(xgb_model, X_train, topk)
    lgb_top_features = get_top_features_from_model(lgb_model, X_train, topk)
    cat_top_features = get_top_features_from_model(cat_model, X_train, topk)

    # Combine the feature importance results from the three models (using the union of features).
    combined_top_features = set(xgb_top_features + lgb_top_features + cat_top_features)

    # Extract SHAP value-based features from each model.
    xgb_shap_top_features = get_shap_top_features(xgb_model, X_train, topk)
    lgb_shap_top_features = get_shap_top_features(lgb_model, X_train, topk)
    cat_shap_top_features = get_shap_top_features(cat_model, X_train, topk)

    # Combine the SHAP value-based features (using the union of features).
    shap_top_features = set(xgb_shap_top_features + lgb_shap_top_features + cat_shap_top_features)

    # Final feature set: the union of model feature importance and SHAP value-based features.
    selected_features = list(combined_top_features.union(shap_top_features))

    # print selected results
    logger.info(f"Selected {len(selected_features)} features out of {total_features} total features.")
    logger.info(f"Selected features: {selected_features}")

    return sorted(selected_features)

def feature_select_rf(X_train, y_train, X_val, y_val, topk_ratio=None):
    '''
    Feature Selection Method Based on Random Forest
    '''
    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_select_rf)
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    selected_features = feature_importance['feature'].iloc[:topk].tolist()
    logger.info(f"Selected {len(selected_features)} features out of {total_features} total features using Random Forest.")
    logger.info(f"Selected features: {selected_features}")
    
    return sorted(selected_features)

def feature_select_chi2(X_train, y_train, X_val, y_val, topk_ratio=None):
    '''
    Feature Selection Method Based on Filter Approach - Chi-Squared Test
    '''
    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_select_chi2)
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))
    
    # Normalize X_train to [0, 1] for chi2
    scaler = MinMaxScaler()
    X_train_normalized = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    
    selector = SelectKBest(score_func=chi2, k=topk)
    selector.fit(X_train_normalized, y_train)
    
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {total_features} total features using Chi-Square Test.")
    logger.info(f"Selected features: {selected_features}")
    
    return sorted(selected_features)

def feature_select_mutual_info(X_train, y_train, X_val, y_val, topk_ratio=None):
    '''
    Feature Selection Method Based on Filter Approach - Mutual Information
    '''
    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_select_mutual_info)
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))
    
    selector = SelectKBest(score_func=mutual_info_regression, k=topk)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {total_features} total features using Mutual Information.")
    logger.info(f"Selected features: {selected_features}")
    
    return sorted(selected_features)

def feature_select_variance_threshold(X_train, y_train, X_val, y_val, topk_ratio=None):
    """
    Feature Selection Based on Filter Approach - Variance Threshold
    """
    selector = VarianceThreshold(threshold=topk_ratio)
    selector.fit(X_train)
    
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]} total features using Variance Threshold.")
    logger.info(f"Selected features: {selected_features}")
    
    return sorted(selected_features)

def feature_select_rfe(X_train, y_train, X_val, y_val, topk_ratio=None):
    """
    Feature Selection Based on Wrapper Approach - Recursive Feature Elimination (RFE)
    """
    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_select_rfe)
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))
    
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=topk, step=1)
    selector = selector.fit(X_train.iloc[:len(X_train)//300, :], y_train.iloc[:len(X_train)//300, :])
    
    selected_features = X_train.columns[selector.support_]
    logger.info(f"Selected {len(selected_features)} features out of {total_features} total features using RFE.")
    logger.info(f"Selected features: {selected_features}")
    
    return sorted(selected_features)

def feature_select_genetic(X_train, y_train, X_val, y_val, topk_ratio=None):
    """
    Feature Selection Based on Wrapper Approach - Genetic Algorithm
    """
    # Adaptive selection of topk_ratio if not provided
    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_select_genetic)
    
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))  # Ensure at least 1 feature is selected

    # Define the estimator (SVR with linear kernel)
    estimator = SVR(kernel="linear")

    try:
        # Initialize GeneticSelectionCV with compatible parameters
        selector = GeneticSelectionCV(
            estimator=estimator,
            cv=5,  # Cross-validation folds
            verbose=0,  # Disable verbose output
            scoring="neg_mean_squared_error",  # Scoring metric
            max_features=topk,  # Maximum number of features to select
            n_population=3,  # Population size for genetic algorithm
            crossover_proba=0.5,  # Crossover probability
            mutation_proba=0.2,  # Mutation probability
            n_generations=5,  # Number of generations
            crossover_independent_proba=0.5,  # Independent crossover probability
            mutation_independent_proba=0.05,  # Independent mutation probability
            tournament_size=2,  # Tournament size for selection
            n_gen_no_change=3,  # Stop if no improvement for 3 generations
            caching=True,  # Enable caching for performance
            n_jobs=-1  # Use all available CPU cores
        )

        # Fit the selector to the training data
        # Explicitly avoid passing fit_params by wrapping the fit call
        selector.fit(X_train.iloc[:len(X_train)//500, :], y_train.iloc[:len(X_train)//500, :])

        # Extract selected features
        selected_features = X_train.columns[selector.support_]

        # Log the results
        logger.info(f"Selected {len(selected_features)} features out of {total_features} total features using Genetic Algorithm.")
        logger.info(f"Selected features: {selected_features}")

    except TypeError as e:
        logger.error(f"TypeError occurred during GeneticSelectionCV fit: {e}")
        raise ValueError("Ensure that the GeneticSelectionCV library is compatible with your scikit-learn version.") from e

    return sorted(selected_features)

def feature_select_lasso(X_train, y_train, X_val, y_val, topk_ratio=None):
    """
    Feature Selection Based on Embedded Method - L1 Regularization (Lasso)
    """
    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_select_lasso)
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))
    
    lasso = Lasso(alpha=0.001)
    selector = SelectFromModel(lasso, prefit=False, max_features=topk)
    selector.fit(X_train.iloc[:len(X_train)//2, :], y_train.iloc[:len(X_train)//2, :])
    
    selected_features = X_train.columns[selector.get_support()]
    logger.info(f"Selected {len(selected_features)} features out of {total_features} total features using Lasso.")
    logger.info(f"Selected features: {selected_features}")
    
    return sorted(selected_features)

def feature_select_lgb_shap(X_train, y_train, X_val, y_val, topk_ratio=None):
    """
    Feature Selection Based on SHAP Values from a Single LGB Model
    """
    if topk_ratio is None:
        topk_ratio = adaptive_topk_ratio_selection(X_train, y_train, X_val, y_val, feature_select_lgb_shap)
    total_features = X_train.shape[1]
    topk = max(1, int(total_features * topk_ratio))
    
    lgb_model = lgb.LGBMRegressor(n_jobs=-1)
    lgb_model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_train)
    shap_sum = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': shap_sum
    }).sort_values(by='importance', ascending=False)
    
    selected_features = feature_importance['feature'].iloc[:topk].tolist()
    logger.info(f"Selected {len(selected_features)} features out of {total_features} total features using LightGBM SHAP.")
    logger.info(f"Selected features: {selected_features}")
    
    return sorted(selected_features)

