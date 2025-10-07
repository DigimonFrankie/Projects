class MachineLearningPipeline:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cat_cols: List[str],
        target_col: str,
        log_cols: Optional[List[str]] = None,
        n_splits: int = 5,
        n_iter: int = 20,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
        shuffle: bool = True,
        scoring: str = "neg_root_mean_squared_error",   # for regression
    ):
        self.dataframe = dataframe
        self.cat_cols = cat_cols or []
        self.log_cols = log_cols or []
        self.target_col = target_col

        self.test_size = test_size
        self.random_state = random_state
        self.stratify_flag = stratify
        self.shuffle = shuffle
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.scoring = scoring

        self.X_train = self.X_test = self.y_train = self.y_test = None

    def _is_classification(self, y: pd.Series, threshold: int = 20) -> bool:
        """Heuristic: classification if non-numeric OR low-cardinality integers."""
        if not pd.api.types.is_numeric_dtype(y):
            return True
        return pd.api.types.is_integer_dtype(y) and y.nunique() <= threshold

    def split_data(self):
        X = self.dataframe.drop(columns=[self.target_col])
        y = self.dataframe[self.target_col]

        stratify_arg = y if (self.stratify_flag and self._is_classification(y)) else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
            shuffle=self.shuffle,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _make_preprocessor(self) -> ColumnTransformer:
        # split numeric columns into log-transformed vs plain
        num_cols = list(set(self.dataframe.columns) - set(self.cat_cols)-set([self.target_col]))
        plain_num = list(set(num_cols) - set(self.log_cols))

        # log1p for selected skewed columns (clip negatives to 0 just in case)
        log_tf = FunctionTransformer(lambda X: np.log1p(np.clip(X, a_min=0, a_max=None)))

        num_steps_log = Pipeline([
            ("log1p", log_tf),
            ("scale", StandardScaler())
        ])

        num_steps_plain = Pipeline([
            ("scale", StandardScaler())
        ])

        # NOTE: use sparse=False for sklearn <1.2 ; use sparse_output=False for >=1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_log", num_steps_log, self.log_cols),
                ("num_plain", num_steps_plain, plain_num),
                ("cat", ohe, self.cat_cols),
            ],
            remainder="drop",
            n_jobs=None
        )
        return preprocessor

    def linear_model_pipeline(self, model: str = "ridge", log_target: bool = False):
        """
        model: 'ridge' or 'elasticnet'
        """
        if any(v is None for v in (self.X_train, self.y_train)):
            raise ValueError("Call split_data() before training.")

        preprocessor = self._make_preprocessor()

        if model == "ols":
            est = LinearRegression()
            SearchCV = GridSearchCV
            param_dist = {
                "est__fit_intercept": [True, False]
                }
        elif model == "ridge":
            est = Ridge(random_state=self.random_state)
            SearchCV = RandomizedSearchCV
            param_dist = {
                "est__alpha": loguniform(1e-3, 1e3)
            }
        elif model == "lasso":
            est = Lasso(random_state=self.random_state, max_iter=5000)
            SearchCV = RandomizedSearchCV
            param_dist = {
                "est__alpha": loguniform(1e-4, 1e2),
            }
        elif model == "elasticnet":
            est = ElasticNet(random_state=self.random_state, max_iter=5000)
            SearchCV = RandomizedSearchCV
            param_dist = {
                "est__alpha": loguniform(1e-4, 1e2),
                "est__l1_ratio": uniform(0, 1)
            }
        else:
            raise ValueError("model must be 'ols', 'ridge', 'lasso' or 'elasticnet'.")

        ## optionally wrap target in log-transform
        if log_target:
            log_t = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
            est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
            
            """
            When wrapping, the param names shift under 'est__regressor__*'
            Only needed if tuning the wrapped regressor (not OLS fit_intercept).
            """ 
            if model != 'ols':
                param_dist = {f"est__regressor__{k.split('__', 1)[1]}": v for k,v in param_dist.items()}
            else:
                param_dist = {"est__regressor__fit_intercept":[True, False]}

        pipe = Pipeline([
            ("prep", preprocessor),
            ("est", est)
        ])

        # KFold for regression
        cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        if model == "ols":
            SearchCV = GridSearchCV
            search_kwargs = {"param_grid": param_dist}
        elif model in ["ridge", "lasso", "elasticnet"]:
            SearchCV = RandomizedSearchCV
            search_kwargs = {
                "param_distributions": param_dist,
                "n_iter": self.n_iter,
                "random_state": self.random_state,
            }
        else:
            raise ValueError("model must be 'ols', 'ridge', 'lasso', or 'elasticnet'.")

        search = SearchCV(
            estimator=pipe,
            scoring=self.scoring,
            cv=cv,
            n_jobs=-1,
            verbose=0,
            **search_kwargs
        )
        

        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        
        print("Best params:", search.best_params_)
        print("CV best score:", search.best_score_)

        return search, y_pred
    
    def compare_linear_models(
        self,
        models=("ridge", "lasso", "elasticnet"),
        log_target: bool = False,
    ):
        """
        Train/compare several linear models with your current pipeline settings.
        Returns: (results_df, models_dict)
            - results_df: one row per model with CV score and test metrics
            - models_dict: {model_name: fitted RandomizedSearchCV}
        """
        results = []
        models_dict = {}

        for m in models:
            search, y_pred = self.linear_model_pipeline(model=m, log_target=log_target)

            # CV score in pipeline is "neg_*" => flip sign for readability
            cv_score = search.best_score_
            cv_metric_name = self.scoring
            if cv_metric_name.startswith("neg_"):
                cv_readable = -cv_score
                cv_metric_name = cv_metric_name.replace("neg_", "")
            else:
                cv_readable = cv_score

            rmse = mean_squared_error(self.y_test, y_pred, squared=False)
            mae  = mean_absolute_error(self.y_test, y_pred)
            r2   = r2_score(self.y_test, y_pred)

            results.append({
                "model": m,
                "log_target": log_target,
                f"cv_{cv_metric_name}": cv_readable,
                "test_RMSE": rmse,
                "test_MAE": mae,
                "test_R2": r2,
                "best_params": search.best_params_,
            })
            models_dict[m] = search

        results_df = pd.DataFrame(results).sort_values(by="test_RMSE").reset_index(drop=True)
        print("\nLeaderboard (lower RMSE is better):")
        display_cols = ["model", "log_target", f"cv_{cv_metric_name}", "test_RMSE", "test_MAE", "test_R2"]
        print(results_df[display_cols].to_string(index=False))
        return results_df, models_dict