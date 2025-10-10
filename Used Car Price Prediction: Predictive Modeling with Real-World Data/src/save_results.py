import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, r2_score

def save_grid_search_results(
    grid_search_func,
    X_train, y_train, X_test, y_test,
    path="../src/", out_file="grid_results.csv"
):
    """
    Runs grid search, saves best params to CSV and best model to file.
    Args:
        grid_search_func: function that returns a fitted GridSearchCV/RandomizedSearchCV object.
        X_train, y_train, X_test, y_test: data.
        path: directory for results/model files (default: './src/')
        out_file: CSV file for results (saved inside path).
    Returns:
        The fitted GridSearchCV object.
    """
    start_time = datetime.now()

    print("Starting training...")
    print("=" * 70)
    grid = grid_search_func(X_train, y_train)

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training completed in {training_time}")
    print("=" * 70)

    print("Start model diagnostics...")
    print("=" * 70)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.3f}")

    results = {
        'date': start_time.strftime('%Y-%m-%d'),
        'time': start_time.strftime('%H:%M:%S'),
        'training_time': str(training_time),
        'best_score': -grid.best_score_,
        **grid.best_params_,
        'test_RMSE': rmse,
        'test_R2': r2,
    }

    # Save results to CSV
    full_result_path = os.path.join(path, out_file)
    os.makedirs(path, exist_ok=True)
    try:
        df = pd.read_csv(full_result_path)
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([results])
    df.to_csv(full_result_path, index=False)
    print("=" * 70)
    print(f"Saved best params for {type(best_model['model']).__name__} to {full_result_path}")

    # Save best model to folder
    model_dir = os.path.join(path, "best_models")
    os.makedirs(model_dir, exist_ok=True)
    model_type = type(best_model['model']).__name__
    model_name = f"{model_type}_{start_time.strftime('%Y%m%d_%H%M%S')}.joblib"
    full_model_path = os.path.join(model_dir, model_name)
    joblib.dump(best_model, full_model_path)
    print("=" * 70)
    print(f"Saved best model as {full_model_path}")

    return grid