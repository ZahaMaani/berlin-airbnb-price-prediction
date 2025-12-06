import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib

class AirbnbEngineer:
    def __init__(self, df):
        self.df = df
        self.columns_to_transform =[
            'minimum_nights', 'maximum_nights', 'estimated_revenue_l365d',
            'reviews_per_month', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d'
        ]
        self.numeric_cols = [
            'host_response_rate', 'host_acceptance_rate', 'availability_30', 'availability_60',
            'availability_90', 'availability_365', 'estimated_occupancy_l365d', 'review_scores_rating',
            'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
            'review_scores_communication', 'review_scores_location', 'review_scores_value',
            'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes',
            'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
            'amenities', 'host_response_time'
       ]
        
    
    def plot_price_distribution(self):
        #Plots the distribution of price and log_price.
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['price'], bins=50, log=True)
        plt.xlabel('Price')
        plt.ylabel('Count (log scale)')
        plt.title('Original Price Distribution')
    
        # Temporary calculation for visualization if log_price doesn't exist yet
        log_price = np.log1p(self.df['price'])
        
        plt.subplot(1, 2, 2)
        plt.hist(log_price, bins=50)
        plt.xlabel('Log Price')
        plt.ylabel('Count')
        plt.title('Log Price Distribution')
        
        plt.show()
    
    def process_target_variable(self, series):
        return np.log1p(series)

    def apply_log_transformations(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col] = np.log1p(self.df[col])
    
    def convert_booleans(self, columns):
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)
    
    def encode_ordinals(self, series):
        mapping = {
            'within a day': 2,
            'a few days or more': 1,
            'within an hour': 4,
            'within a few hours': 3
        }
        return series.map(mapping)

    def encode_categoricals(self, columns):
        df = self.df.copy()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        encoded_data = encoder.fit_transform(df[columns])
        feature_names = encoder.get_feature_names_out(columns)
        
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
        self.df = pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)
    
    def engineer_amenities(self, series):
        return series.astype(str).str.count(',') + 1

    def new_neighborhood(self):
        df = self.df.copy()
        return df['neighbourhood_group_cleansed'] + '_' + df['neighbourhood_cleansed']
    
    def split_dataset(self, target_col='log_price', test_size=0.2, random_state=42):
        df = self.df.copy()
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_numeric_features(self, X_train, X_test):
        scaler = MinMaxScaler()
        
        scaler.fit(X_train[self.numeric_cols])
        X_train[self.numeric_cols] = scaler.transform(X_train[self.numeric_cols])
        X_test[self.numeric_cols] = scaler.transform(X_test[self.numeric_cols])
        
        return X_train, X_test
    
    def target_encode_neighbourhood(self, X_train, X_test, y_train):
        # Calculate target mean on training set
        temp_train = X_train.copy()
        temp_train['log_price'] = y_train
        neigh_mean = temp_train.groupby('full_neighbourhood')['log_price'].mean()
        global_mean = y_train.mean()
        
        # Map to Train and Test
        X_train['full_neighbourhood_target'] = X_train['full_neighbourhood'].map(neigh_mean)
        
        # Map to Test (fill new/unknown neighbourhoods with global mean)
        X_test['full_neighbourhood_target'] = X_test['full_neighbourhood'].map(neigh_mean).fillna(global_mean)
        
        # Drop original columns
        cols_to_drop = ['neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'full_neighbourhood']
        X_train.drop(cols_to_drop, axis=1, inplace=True)
        X_test.drop(cols_to_drop, axis=1, inplace=True)

        print('Train data prepared.')
        
        return X_train, X_test

    def transform(self):
        #Apply log transformation for the price column for skewness
        self.df['log_price'] = self.process_target_variable(self.df['price'])
        self.df.drop(columns = 'price', errors='ignore', inplace = True)

        #Apply log transformation for other numerical column as needed
        self.apply_log_transformations(self.columns_to_transform)

        #Convert True/False to 0/1
        columns_to_boolean = ['instant_bookable', 'has_availability']
        self.convert_booleans(columns_to_boolean)

        #Encoding host_respones_time column (treated as ordinal values)
        self.df['host_response_time'] = self.encode_ordinals(self.df['host_response_time'])

        #One hot encoding for categorical columns
        cols_to_encode = ['property_type', 'room_type']
        self.encode_categoricals(cols_to_encode)

        #Count the number of amenities
        self.df['amenities'] = self.engineer_amenities(self.df['amenities'])

        #Create new full_neighborhood column
        self.df['full_neighbourhood'] = self.new_neighborhood()

        print('Data encoded')
        print(self.df.info())

    def save_preprocessor(self, filepath):
        #Saves this entire class object to a file
        joblib.dump(self, filepath)

    @staticmethod
    def load_preprocessor(filepath):
        #Loads the class object back into memory
        return joblib.load(filepath)