import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, Dict, List

def drop_column(raw_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Drop a specified column from the DataFrame.

    Args:
        raw_df (pd.DataFrame): The input DataFrame from which to drop the column.
        column (str): The name of the column to drop.

    Returns:
        pd.DataFrame: The DataFrame with the specified column dropped.
    """
    raw_df = raw_df.drop(columns=[column])
    return raw_df

def split_df(raw_df: pd.DataFrame, target: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the DataFrame into training and validation sets.

    Args:
        raw_df (pd.DataFrame): The input DataFrame containing the features and target.
        target (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the validation set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            - X_train: Training feature set.
            - X_val: Validation feature set.
            - train_targets: Training target values.
            - val_targets: Validation target values.
    """
    X = raw_df.drop(target, axis=1)
    y = raw_df[target]
    
    X_train, X_val, train_targets, val_targets = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    return X_train, X_val, train_targets, val_targets

def numeric_categorical_cols(X_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns in the training set.

    Args:
        X_train (pd.DataFrame): The training feature set.

    Returns:
        Tuple[List[str], List[str]]: 
            - List of numeric column names.
            - List of categorical column names.
    """
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes('object').columns.tolist()
    return numeric_cols, categorical_cols

def scale_numerical(X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale numeric columns in the training and validation sets using Min-Max scaling.

    Args:
        X_train (pd.DataFrame): The training feature set.
        X_val (pd.DataFrame): The validation feature set.
        numeric_cols (List[str]): The names of the numeric columns to scale.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Scaled training feature set.
            - Scaled validation feature set.
    """
    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val

def encoder_categorical(X_train: pd.DataFrame, X_val: pd.DataFrame, categorical_cols: List[str], drop_original: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical columns in the training and validation sets using One-Hot encoding.

    Args:
        X_train (pd.DataFrame): The training feature set.
        X_val (pd.DataFrame): The validation feature set.
        categorical_cols (List[str]): The names of the categorical columns to encode.
        drop_original (bool): Whether to drop original categorical columns after encoding.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Encoded training feature set.
            - Encoded validation feature set.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    X_train[encoded_cols] = encoder.transform(X_train[categorical_cols])
    X_val[encoded_cols] = encoder.transform(X_val[categorical_cols])
    
    if drop_original:
        X_train = X_train.drop(columns=categorical_cols)
        X_val = X_val.drop(columns=categorical_cols)
    
    return X_train, X_val


def preprocess_data(
    raw_df: pd.DataFrame, 
    column_to_drop: str, 
    target: str, 
    test_size: float, 
    scaler_numeric: bool = True
):
    """
    Preprocess the raw DataFrame by dropping specified columns, splitting the data,
    and scaling/encoding the features.

    Args:
        raw_df (pd.DataFrame): The input DataFrame to preprocess.
        column_to_drop (str): The column to drop from the DataFrame.
        target (str): The target column name.
        test_size (float): The proportion of the dataset to include in the validation set.
        scaler_numeric (bool): Whether to scale numeric columns.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, object]]: 
            A tuple containing:
            - A dictionary with 'X_train', 'X_val', 'train_targets', and 'val_targets'.
            - A dictionary containing the fitted scaler and encoder.
    """
    raw_df = drop_column(raw_df, column_to_drop)
    X_train, X_val, train_targets, val_targets = split_df(raw_df, target, test_size)
    numeric_cols, categorical_cols = numeric_categorical_cols(X_train)

    # Initialize scaler and encoder
    scaler = MinMaxScaler() if scaler_numeric else None
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Scale numeric columns if required
    if scaler_numeric:
        X_train, X_val = scale_numerical(X_train, X_val, numeric_cols)
    
    # Encode categorical columns
    X_train, X_val = encoder_categorical(X_train, X_val, categorical_cols, drop_original=True)

    return (
        {
            'X_train': X_train,
            'X_val': X_val,
            'train_targets': train_targets,
            'val_targets': val_targets
        },
        {
            'scaler': scaler,
            'encoder': encoder
        }
    )

def preprocess_new_data(
    new_data: pd.DataFrame, 
    fitted_transformers: Dict,
    column_to_drop: str
):
    """
    Preprocess new data for prediction by dropping specified columns, scaling numeric features, 
    and encoding categorical features using fitted transformers.

    Args:
        new_data (pd.DataFrame): The new data to preprocess.
        fitted_transformers (Dict): A dictionary containing fitted transformers for scaling and encoding.
        column_to_drop (str): The column to drop from the dataset.

    Returns:
        pd.DataFrame: The preprocessed feature set (DataFrame).
    """
    raw_df = new_data.drop(columns=[column_to_drop])
    
    X = raw_df
    
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    

    if 'scaler' in fitted_transformers:
        scaler = fitted_transformers['scaler']
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    

    if 'encoder' in fitted_transformers:
        encoder = fitted_transformers['encoder']
        X_encoded = encoder.fit_transform(X[categorical_cols])
        
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        
        X = pd.concat([X.drop(columns=categorical_cols), X_encoded_df], axis=1)

    return X


