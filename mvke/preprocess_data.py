import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import argparse

def load_movielens_data(data_path='data/ml-100k'):
    """Load MovieLens 100K dataset"""
    # Load ratings
    ratings = pd.read_csv(
        os.path.join(data_path, 'u.data'), 
        sep='\t', 
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    # Load user features
    users = pd.read_csv(
        os.path.join(data_path, 'u.user'), 
        sep='|', 
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )
    
    # Load item features
    items = pd.read_csv(
        os.path.join(data_path, 'u.item'), 
        sep='|', 
        encoding='latin-1',
        names=['item_id', 'title', 'release_date', 'video_release_date', 
               'imdb_url'] + [f'genre_{i}' for i in range(19)]
    )
    
    return ratings, users, items

def create_synthetic_labels(ratings):
    """Create synthetic CTR and CVR labels based on ratings"""
    # CTR: probability based on whether user rated the item
    ratings['ctr_label'] = 1  # All samples in dataset are positive for CTR
    
    # CVR: probability based on high rating (4 or 5)
    ratings['cvr_label'] = (ratings['rating'] >= 4).astype(int)
    
    return ratings

def create_negative_samples(ratings, users, items, neg_ratio=1):
    """Create negative samples for training"""
    print("Creating negative samples...")
    all_user_ids = users['user_id'].unique()
    all_item_ids = items['item_id'].unique()
    
    # Create set of positive samples
    positive_set = set(zip(ratings['user_id'], ratings['item_id']))
    
    negative_samples = []
    for _, row in ratings.iterrows():
        user_id = row['user_id']
        # Sample negative items for this user
        for _ in range(neg_ratio):
            neg_item = np.random.choice(all_item_ids)
            while (user_id, neg_item) in positive_set:
                neg_item = np.random.choice(all_item_ids)
            
            negative_samples.append({
                'user_id': user_id,
                'item_id': neg_item,
                'rating': 0,
                'timestamp': row['timestamp'],
                'ctr_label': 0,
                'cvr_label': 0
            })
    
    neg_df = pd.DataFrame(negative_samples)
    ratings_with_neg = pd.concat([ratings, neg_df], ignore_index=True)
    
    return ratings_with_neg

def process_features(ratings, users, items):
    """Process and encode features"""
    print("Processing features...")
    
    # Merge all data
    data = ratings.merge(users, on='user_id').merge(items, on='item_id')
    
    # Process user features
    le_gender = LabelEncoder()
    le_occupation = LabelEncoder()
    
    data['gender_encoded'] = le_gender.fit_transform(data['gender'])
    data['occupation_encoded'] = le_occupation.fit_transform(data['occupation'])
    
    # Process item features - genres
    genre_columns = [f'genre_{i}' for i in range(19)]
    
    # Calculate some aggregate features
    user_stats = ratings.groupby('user_id').agg({
        'rating': ['count', 'mean'],
        'item_id': 'nunique'
    }).reset_index()
    user_stats.columns = ['user_id', 'user_rating_count', 'user_rating_mean', 'user_item_count']
    
    item_stats = ratings.groupby('item_id').agg({
        'rating': ['count', 'mean'],
        'user_id': 'nunique'
    }).reset_index()
    item_stats.columns = ['item_id', 'item_rating_count', 'item_rating_mean', 'item_user_count']
    
    # Merge stats
    data = data.merge(user_stats, on='user_id', how='left')
    data = data.merge(item_stats, on='item_id', how='left')
    
    # Fill NaN values
    data = data.fillna(0)
    
    return data

def create_feature_columns():
    """Define feature columns for the model"""
    user_features = [
        'user_id', 'age', 'gender_encoded', 'occupation_encoded',
        'user_rating_count', 'user_rating_mean', 'user_item_count'
    ]
    
    item_features = [
        'item_id', 'item_rating_count', 'item_rating_mean', 'item_user_count'
    ] + [f'genre_{i}' for i in range(19)]
    
    return user_features, item_features

def split_data(data, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets"""
    print("Splitting data...")
    
    # Sort by timestamp for temporal split
    data_sorted = data.sort_values('timestamp')
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(data_sorted)), 
        test_size=test_size, 
        shuffle=False
    )
    
    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=val_size/(1-test_size), 
        shuffle=False
    )
    
    train_data = data_sorted.iloc[train_idx]
    val_data = data_sorted.iloc[val_idx]
    test_data = data_sorted.iloc[test_idx]
    
    return train_data, val_data, test_data

def save_processed_data(train_data, val_data, test_data, output_dir='data/processed'):
    """Save processed data"""
    print("Saving processed data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle files
    train_data.to_pickle(os.path.join(output_dir, 'train.pkl'))
    val_data.to_pickle(os.path.join(output_dir, 'val.pkl'))
    test_data.to_pickle(os.path.join(output_dir, 'test.pkl'))
    
    # Save feature information
    user_features, item_features = create_feature_columns()
    feature_info = {
        'user_features': user_features,
        'item_features': item_features,
        'num_users': train_data['user_id'].max() + 1,
        'num_items': train_data['item_id'].max() + 1,
        'num_occupations': train_data['occupation_encoded'].max() + 1
    }
    
    with open(os.path.join(output_dir, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"CTR positive ratio in train: {train_data['ctr_label'].mean():.3f}")
    print(f"CVR positive ratio in train: {train_data['cvr_label'].mean():.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/ml-100k', help='Path to MovieLens data')
    parser.add_argument('--output_dir', default='data/processed', help='Output directory')
    parser.add_argument('--neg_ratio', type=int, default=1, help='Negative sampling ratio')
    args = parser.parse_args()
    
    # Load data
    ratings, users, items = load_movielens_data(args.data_path)
    
    # Create synthetic labels
    ratings = create_synthetic_labels(ratings)
    
    # Create negative samples
    ratings = create_negative_samples(ratings, users, items, args.neg_ratio)
    
    # Process features
    data = process_features(ratings, users, items)
    
    # Split data
    train_data, val_data, test_data = split_data(data)
    
    # Save processed data
    save_processed_data(train_data, val_data, test_data, args.output_dir)
    
    print("Data preprocessing completed!")

if __name__ == '__main__':
    main()