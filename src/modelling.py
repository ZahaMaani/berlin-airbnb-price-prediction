import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class AirbnbModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.rf_model = None
        self.predictions = {}

    def random_forest(self):
        # Train Random Forest with tuned parameters
        
        rf_final = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
        rf_final.fit(self.X_train, self.y_train)
        self.rf_model = rf_final
        self.predictions['rf'] = rf_final.predict(self.X_test)
        print("Random Forest trained.")
        return rf_final

    def evaluation_metrics(self, model_key='rf'):
        if model_key not in self.predictions:
            print("Model not found.")
            return
        pred = self.predictions[model_key]
        mae = mean_absolute_error(self.y_test, pred)
        rmse = mean_squared_error(self.y_test, pred) ** 0.5
        r2 = r2_score(self.y_test, pred)
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def plot_feature_importance(self, n_features=20):
        #Plotting the most important features
        
        if self.rf_model is None: return
        importances = self.rf_model.feature_importances_
        feat_imp = pd.DataFrame({'feature': self.X_train.columns, 'importance': importances})
        feat_imp = feat_imp.sort_values(by='importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feat_imp.head(n_features))
        plt.title("Feature Importance")
        plt.show()

    def plot_residuals(self, model_key='rf'):
        #Resduals plotting
        
        if model_key not in self.predictions: return
        
        pred = self.predictions[model_key]
        residuals = self.y_test - pred
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.show()

    def gradient_boosting(self):
        # Train Gradient Boosting - for camparison reasons
        
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(self.X_train, self.y_train)
        self.predictions['gbr'] = gbr.predict(self.X_test)
        self.evaluation_metrics(model_key='gbr')
        return gbr

    def save_model(self, path='../models/rf_airbnb_final.pkl'):
        # Saving trained Random Forest
        
        if self.rf_model:
            joblib.dump(self.rf_model, path, compress = 3)
            print(f"Saved to {path}")

    @staticmethod
    def load_model(filepath):
        #Loads the class object back into memory
        return joblib.load(filepath)