import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class LinearRegressionEstimator:
    def __init__(
        self,
        outcome_name: str = "outcome",
        group_name: str = "group"
    ):
        self.outcome_name = outcome_name
        self.group_name = group_name

    def linear_regression(self, y: pd.DataFrame, X: pd.DataFrame):
        """Perform linear regression and return the estimated treatment effect coefficient and p-value."""
        model = sm.OLS(y, X)
        results = model.fit()
        return results.params["group"], results.pvalues["group"]

    def calculate_results(self, individ_data, additional_feature_names: list[str] = None):
        """Compute linear regression results across simulations and plot estimated vs. true effects."""
        y = individ_data[self.outcome_name]
        if additional_feature_names:
            X = individ_data[additional_feature_names + [self.group_name]]
        else:
            X = individ_data[self.group_name]

        return self.linear_regression(y, X)


class PropensityScoreMatcher:
    def __init__(
        self,
        treatment_col: str = "group",
        caliper: float = 0.05,
        k: int = 5
    ):
        self.treatment_col = treatment_col
        self.caliper = caliper
        self.k = k

    def match(self, df: pd.DataFrame, covariate_cols: list) -> pd.DataFrame:
        X = df[covariate_cols]
        y = df[self.treatment_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(solver='liblinear')
        model.fit(X_scaled, y)

        df['propensity_score'] = model.predict_proba(X_scaled)[:, 1]

        treated = df[df[self.treatment_col] == 1].copy()
        control = df[df[self.treatment_col] == 0].copy()

        nn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn.fit(control[['propensity_score']])

        distances, indices = nn.kneighbors(treated[['propensity_score']])

        matched_indices = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist[0] < self.caliper:
                matched_indices.append(control.index[idx[0]])

        matched_control = control.loc[matched_indices]
        df_matched = pd.concat([treated, matched_control])

        return df_matched
