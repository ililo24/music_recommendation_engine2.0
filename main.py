import pandas as pd
from data_preparation import load_and_prepare_data
from feature_engineering import *
from model_training import *

def run_pipeline():
    """Run the complete music recommendation pipeline"""
    print("🎵 Starting Music Recommendation Pipeline...")
   
    print("📊 Step 1: Loading data...")
    train_df = load_and_prepare_data()

    print("🔧 Step 2: Feature engineering...")
    train_df = create_time_features(train_df)
    train_df = create_categorical_features(train_df)
    train_df = create_completion_features(train_df)
    train_df, track_stats = create_engagement_features(train_df)
    train_df = calculate_preference_score(train_df)
    
    print("🤖 Step 3: Training model...")
    model = train_model(train_df)

    train_df.to_csv('data/processed_data.csv', index=False)

    print("✅ Pipeline completed successfully!")
    return model, train_df

if __name__ == "__main__":
    model, data = run_pipeline()
