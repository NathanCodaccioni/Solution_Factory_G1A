import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV file, create the brightness feature and assign true labels.
    
    Assumes that the first 20 rows are 'clean' and the next 20 are 'dirty'.
    """
    df = pd.read_csv(filepath)
    # Create a new brightness feature as the average of r, g, b
    df['brightness'] = (df['avg_r'] + df['avg_g'] + df['avg_b']) / 3
    
    # Manually assign labels on the first 40 rows
    df.loc[:19, 'true_label'] = 'clean'
    df.loc[20:39, 'true_label'] = 'dirty'
    return df

def get_numeric_features(df: pd.DataFrame, exclude: list = ['width', 'height']) -> list:
    """
    Retrieve numeric columns excluding those that are not useful.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    return numeric_cols

def compute_feature_stats(df: pd.DataFrame, numeric_cols: list, label_col: str = 'true_label') -> dict:
    """
    For each numeric feature, compute the discriminative power as |Δmean|/std over the first 40 rows.
    
    The threshold is defined as the average of the means for the 'clean' and 'dirty' groups.
    The 'direction' indicates which group has the higher mean.
    """
    stats = {}
    labeled = df.loc[:39]  # use only the labeled subset
    for col in numeric_cols:
        clean_vals = labeled[labeled[label_col] == 'clean'][col]
        dirty_vals = labeled[labeled[label_col] == 'dirty'][col]
        std = labeled[col].std(ddof=0)
        if std == 0:
            continue
        d_mean = abs(clean_vals.mean() - dirty_vals.mean())
        score = d_mean / std
        stats[col] = {
            'score': score,
            'threshold': (clean_vals.mean() + dirty_vals.mean()) / 2,
            'std': std,
            'direction': 'clean' if clean_vals.mean() > dirty_vals.mean() else 'dirty'
        }
    return stats

class RuleBasedClassifier:
    """
    A rule-based classifier that uses weighted z-score logic.
    
    For each top feature the classifier computes the z-score relative to the feature's threshold,
    adjusts the sign if a higher feature value is associated with 'dirty', and sums the weighted z-scores.
    A final decision is made based on whether the total exceeds the `delta` threshold.
    """
    def __init__(self, feature_stats: dict, top_n: int = 20, delta: float = 0.5):
        self.feature_stats = feature_stats
        self.top_n = top_n
        self.delta = delta
        # Select the top N features sorted by their discriminative score
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['score'], reverse=True)
        self.top_features = [feat for feat, _ in sorted_features[:top_n]]
    
    def classify_row(self, row: pd.Series) -> str:
        total = 0.0
        for feat in self.top_features:
            stat = self.feature_stats[feat]
            t = stat['threshold']
            std = stat['std']
            weight = stat['score']
            # Compute the z-score
            z = (row[feat] - t) / std if std != 0 else 0
            # Adjust z if the higher value indicates 'dirty'
            if stat['direction'] == 'dirty':
                z = -z
            total += weight * z
        return 'clean' if total > self.delta else 'dirty'
    
    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply classification to every row in the DataFrame.
        """
        return df.apply(self.classify_row, axis=1)

def main():
    # Load the dataset
    df = load_data("features.csv")
    
    # Get numeric features from the DataFrame
    numeric_cols = get_numeric_features(df, exclude=['width', 'height'])  # adjust exclusions as needed
    print(f"Number of numeric features considered: {len(numeric_cols)}")
    
    # Compute per-feature statistics on the labeled subset
    feature_stats = compute_feature_stats(df, numeric_cols)
    
    # Print the top 20 features ranked by discriminative power
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['score'], reverse=True)
    top_features_formatted = [f"{feat} (score={stats['score']:.2f})" for feat, stats in sorted_features[:20]]
    print("Top features ranked by discriminative power:")
    print(top_features_formatted)
    
    # Initialize and apply the rule-based classifier
    classifier = RuleBasedClassifier(feature_stats=feature_stats, top_n=27, delta=0.01)
    df['auto_label'] = classifier.classify(df)
    
    # Save predictions (for further analysis if needed)
    df.to_csv("train_prediction.csv", index=False)
    
   

if __name__ == '__main__':
    # Step 1: Load extracted image features
    # Step 2: Compute useful numeric columns
    # Step 3: Analyze discriminative power of features
    # Step 4: Build and apply rule-based classifier
    # Step 5: Save and evaluate results
    main()
    
