import pandas as pd
import numpy as np
import joblib

class AirbnbPreprocessor:
    def __init__(self):
        self.medians = {}
        self.modes = {}
        #Asuume if there is no review, that the review value is 0
        self.columns_to_zeros = ['reviews_per_month', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                             'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']
        
        self.columns_to_drop = [
            'id', 'listing_url', 'scrape_id', 'source', 'name', 'description', 'neighborhood_overview',
            'picture_url', 'host_id', 'host_since','host_verifications', 'host_url', 
            'host_name', 'host_location', 'host_about', 'host_thumbnail_url', 'host_picture_url',
            'host_is_superhost','host_neighbourhood', 'host_total_listings_count', 'host_has_profile_pic', 
            'neighbourhood', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
            'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 
            'license', 'calendar_last_scraped', 'number_of_reviews_ly', 'availability_eoy',
            'calendar_updated', 'first_review', 'last_review', 'last_scraped', 'longitude', 'latitude'
        ]
        
    def _clean_currency(self, series):
        #Will use this for the price
        return series.astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

    def _clean_bathrooms(self, series):
        #Will use this for the bathrooms column
        return series.str.extract(r'([\d\.]+)').astype(float)

    def _clean_rates(self, series):
        #Will remove the percentage sign from the rates columns
        return series.astype(str).str.rstrip('%').astype(float)

    def _clean_revenue(self, series):
        #Clean the revenue column
        return series.astype(str).replace(r'[$,]', '', regex=True).astype(float)

    
    def fit(self, df):
        temp_df = df.copy()
        
        # We must clean text columns first, otherwise we can't get a median of strings
        if 'price' in temp_df.columns:
            temp_df['price'] = self._clean_currency(temp_df['price'])
            
        if 'host_response_rate' in temp_df.columns:
            temp_df['host_response_rate'] = self._clean_rates(temp_df['host_response_rate'])
            
        if 'host_acceptance_rate' in temp_df.columns:
            temp_df['host_acceptance_rate'] = self._clean_rates(temp_df['host_acceptance_rate'])

        if 'host_acceptance_rate' in temp_df.columns:
            temp_df['estimated_revenue_l365d'] = self._clean_rates(temp_df['estimated_revenue_l365d'])
        
        # 2. Store the median for every numeric column you care about
        self.medians['price'] = temp_df['price'].median()
        self.medians['bedrooms'] = temp_df['bedrooms'].median()
        self.medians['beds'] = temp_df['beds'].median()
        self.medians['bathrooms'] = temp_df['bathrooms'].median()
        self.medians['host_response_rate'] = temp_df['host_response_rate'].median()
        self.medians['host_acceptance_rate'] = temp_df['host_acceptance_rate'].median()
        self.medians['host_listings_count'] = temp_df['host_listings_count'].median()
        self.medians['estimated_revenue_l365d'] = temp_df['estimated_revenue_l365d'].median()

        #3. Store the modes for every categorical value
        columns_mode = ['host_response_time', 'host_is_superhost', 'host_identity_verified', 'has_availability']
        for col in columns_mode:
            if col in temp_df.columns:
                self.modes[col] = temp_df[col].mode()[0]

        print("Training complete.")
        return self

    def transform(self, df):
        """
        Takes a dirty dataframe (training OR single user input from Streamlit)
        and cleans it using the rules learned in 'fit'.
        """
        df_clean = df.copy()
        
        # 1. Drop Columns (Only if they exist in the input)
        df_clean.drop(columns=self.columns_to_drop, errors='ignore', inplace=True)
        
        # 2. Basic Text Cleaning
        if 'price' in df_clean.columns:
            df_clean['price'] = self._clean_currency(df_clean['price'])
        
        if 'bathrooms_text' in df_clean.columns:
            df_clean['bathrooms'] = self._clean_bathrooms(df_clean['bathrooms_text'])
            df_clean.drop(columns=['bathrooms_text'], inplace=True)

        if 'host_response_rate' in df_clean.columns:
            df_clean['host_response_rate'] = self._clean_rates(df_clean['host_response_rate'])

        if 'host_acceptance_rate' in df_clean.columns:
            df_clean['host_acceptance_rate'] = self._clean_rates(df_clean['host_acceptance_rate'])

        if 'host_acceptance_rate' in df_clean.columns:
            df_clean['estimated_revenue_l365d'] = self._clean_rates(df_clean['estimated_revenue_l365d'])
            
        # 3. Fill Missing Values using the learned medians
        for col, value in self.medians.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(value)
        
        for col, value in self.modes.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(value)

        for col in self.columns_to_zeros:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # 4. Handle Booleans
        bool_cols = ['instant_bookable', 'has_availability', 'host_identity_verified']
        for col in bool_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'t': True, 'f': False})
                
        return df_clean

    def save_preprocessor(self, filepath):
        #Saves this entire class object to a file
        joblib.dump(self, filepath)

    @staticmethod
    def load_preprocessor(filepath):
        #Loads the class object back into memory
        return joblib.load(filepath)