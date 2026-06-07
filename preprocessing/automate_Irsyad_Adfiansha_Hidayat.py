"""
automate_Siswa.py
Automated data preprocessing for Heart Disease Dataset.
Konversi dari notebook eksperimen ke script otomatis.
"""

import os
import sys
import argparse
import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(raw_dir: str) -> pd.DataFrame:
    logger.info("Loading local dataset...")

    file_path = "../heart_disease_raw/heart_raw.csv"

    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded: {df.shape}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: median for numeric, mode for categorical."""
    logger.info("Handling missing values...")
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    logger.info("Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    logger.info(f"Rows: {before} -> {after} (removed {before - after} duplicates)")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Handle outliers using IQR clipping on continuous columns."""
    logger.info("Handling outliers with IQR clipping...")
    df = df.copy()
    continuous_cols = [c for c in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                       if c in df.columns]

    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        logger.info(f"  {col}: {n_outliers} outliers clipped")

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical (object) columns."""
    logger.info("Encoding categorical features...")
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not cat_cols:
        logger.info("  No categorical columns found.")
        return df

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        logger.info(f"  Encoded: {col}")

    return df


def scale_features(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """Standardize features using StandardScaler (exclude target)."""
    logger.info("Standardizing features...")
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info(f"  Scaled {len(feature_cols)} features.")
    return df


def preprocess(raw_output_dir: str = "heart_disease_raw",
               preprocessed_output_dir: str = "heart_disease_preprocessed",
               target_col: str = "target") -> pd.DataFrame:
   
    logger.info("=" * 50)
    logger.info("Starting preprocessing pipeline...")
    logger.info("=" * 50)

    # Step 1: Load
    df = load_dataset(raw_output_dir)

    # Step 2: Missing values
    df = handle_missing_values(df)

    # Step 3: Duplicates
    df = remove_duplicates(df)

    # Step 4: Outliers
    df = handle_outliers(df)

    # Step 5: Encoding
    df = encode_categorical(df)

    # Step 6: Scaling
    df = scale_features(df, target_col)

    # Save preprocessed data
    os.makedirs(preprocessed_output_dir, exist_ok=True)
    output_path = os.path.join(preprocessed_output_dir, "heart_preprocessed.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")
    logger.info("=" * 50)
    logger.info("Preprocessing pipeline completed successfully!")
    logger.info("=" * 50)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Heart Disease Data Preprocessing")
    parser.add_argument("--raw_dir", default="heart_disease_raw", help="Raw data output directory")
    parser.add_argument("--preprocessed_dir", default="heart_disease_preprocessed",
                        help="Preprocessed data output directory")
    parser.add_argument("--target_col", default="target", help="Target column name")
    args = parser.parse_args()

    df_preprocessed = preprocess(
        raw_output_dir=args.raw_dir,
        preprocessed_output_dir=args.preprocessed_dir,
        target_col=args.target_col
    )
    print(f"\nPreprocessed DataFrame shape: {df_preprocessed.shape}")
    print(df_preprocessed.head())
