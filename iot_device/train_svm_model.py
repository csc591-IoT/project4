import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DoorStateClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'a_mag', 'w_mag']
        self.class_names = ['Open', 'Close']
        
    def load_training_data_from_csv(self, filename="training_data.csv"):
        """Load training data from CSV file with 2 classes"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Training data file {filename} not found")
        
        # Read CSV file
        df = pd.read_csv(filename)
        
        print(f"CSV file loaded: {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"\nOriginal label distribution:")
        print(df['label'].value_counts())
        
        # Extract features
        feature_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'a_mag', 'w_mag']
        
        # Check if all feature columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        X = df[feature_columns].values
        
        # Map labels to numeric classes
        # open = 0, close = 1
        label_map = {
            'open': 0,
            'opening': 0,
            'close': 1,
            'closed': 1,
            'closing': 1
        }
        
        # Convert labels
        y = df['label'].map(label_map)
        
        # Handle any unmapped labels
        if y.isna().any():
            unmapped = df[y.isna()]['label'].unique()
            print(f"\nWarning: Found unmapped labels: {unmapped}")
            raise ValueError(f"Unmapped labels found: {unmapped}")
        
        y = y.values
        
        print(f"\nMapped label distribution:")
        print(f"Class 0 (Open): {np.sum(y == 0)} samples")
        print(f"Class 1 (Close): {np.sum(y == 1)} samples")
        print(f"\nTotal: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Preprocess the training data"""
        # Remove any samples with NaN or infinite values
        valid_mask = np.isfinite(X).all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"After removing invalid samples: {X.shape[0]} samples")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train SVM model with hyperparameter tuning"""
        # Check if we have enough samples
        if len(X) < 10:
            raise ValueError("Not enough training samples. Need at least 10 samples.")
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass balance:")
        for cls, cnt in zip(unique, counts):
            print(f"  Class {int(cls)} ({self.class_names[int(cls)]}): {cnt} samples ({cnt/len(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Try different SVM parameters
        param_grid = [
            {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
            {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
            {'C': 1.0, 'kernel': 'linear'},
            {'C': 1.0, 'kernel': 'poly', 'degree': 2},
            {'C': 1.0, 'kernel': 'poly', 'degree': 3},
        ]
        
        best_score = 0
        best_params = None
        best_model = None
        
        print("\nTesting different SVM parameters...")
        for params in param_grid:
            model = SVC(**params, random_state=random_state)
            
            # Cross-validation
            cv_folds = min(5, len(X_train) // 2)  # Need at least 2 samples per class
            if cv_folds < 2:
                cv_folds = 2
                
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            print(f"Params: {params}")
            print(f"CV Score: {mean_score:.3f} (+/- {std_score * 2:.3f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = model
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score: {best_score:.3f}")
        
        # Train final model with best parameters
        self.model = best_model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest set accuracy: {test_accuracy:.3f}")
        print("\n" + "="*60)
        print("Classification Report:")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        print("\n" + "="*60)
        print("Confusion Matrix:")
        print("="*60)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"\nRows = True labels, Columns = Predicted labels")
        
        # Calculate per-class accuracy
        print(f"\nPer-class accuracy:")
        for i, class_name in enumerate(self.class_names):
            class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"  {class_name}: {class_acc:.3f} ({cm[i, i]}/{cm[i, :].sum()})")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - 2-Class Door State Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filename = 'confusion_matrix_2class.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {filename}")
        plt.close()
        
        return test_accuracy, best_params

    def save_model(self, model_filename="30_entries_model.pkl", 
                   scaler_filename="30_entries_scaler.pkl"):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        
        print(f"\nModel saved to {model_filename}")
        print(f"Scaler saved to {scaler_filename}")
    
    def load_model(self, model_filename="door_classifier_2class.pkl", 
                   scaler_filename="door_scaler_2class.pkl"):
        """Load trained model and scaler"""
        if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
            raise FileNotFoundError("Model or scaler files not found")
        
        self.model = joblib.load(model_filename)
        self.scaler = joblib.load(scaler_filename)
        
        print(f"Model loaded from {model_filename}")
        print(f"Scaler loaded from {scaler_filename}")
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert numeric predictions to class names
        pred_names = [self.class_names[int(p)] for p in predictions]
        
        return predictions, pred_names
    
    def predict_proba(self, X):
        """Get prediction probabilities (if available)"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded")
        
        # Check if model supports probability
        if not hasattr(self.model, 'predict_proba'):
            print("This model doesn't support probability predictions")
            print("Retrain with probability=True in SVC")
            return None
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def get_feature_importance(self):
        """Get feature importance (for linear kernel)"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        if hasattr(self.model, 'coef_'):
            # For binary linear SVM
            importance = np.abs(self.model.coef_).flatten()
            
            print(f"\nFeature Importance:")
            for i, (feat_name, imp) in enumerate(zip(self.feature_names, importance)):
                print(f"{i+1}. {feat_name}: {imp:.3f}")
            
            return importance
        else:
            print("Feature importance not available for non-linear kernels")
            return None

def analyze_csv_data(filename="new_test_data_with_motion.csv"):
    """Analyze the CSV training data"""
    if not os.path.exists(filename):
        print(f"Training data file {filename} not found")
        return
    
    df = pd.read_csv(filename)
    
    print(f"\n{'='*60}")
    print(f"CSV Training Data Analysis")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nLabel distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print(f"\nLabel percentages:")
    print(df['label'].value_counts(normalize=True) * 100)
    
    # Feature columns
    feature_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'a_mag', 'w_mag']
    
    if all(col in df.columns for col in feature_columns):
        print(f"\n{'='*60}")
        print(f"Feature Statistics:")
        print(f"{'='*60}")
        print(df[feature_columns].describe())
        
        # Check for NaN values
        nan_count = df[feature_columns].isna().sum().sum()
        if nan_count > 0:
            print(f"\nWarning: {nan_count} NaN values found in features")
    else:
        missing = [col for col in feature_columns if col not in df.columns]
        print(f"\nWarning: Missing feature columns: {missing}")

def main():
    import sys
    
    print("\n" + "="*70)
    print("Door State SVM Classifier - 2-Class Classification")
    print("Classes: Open (0), Close (1)")
    print("="*70 + "\n")
    
    # Get CSV filename from command line or use default
    csv_filename = sys.argv[1] if len(sys.argv) > 1 else "data_with_30_entries.csv"
    
    # Analyze CSV data
    analyze_csv_data(csv_filename)
    
    # Initialize classifier
    classifier = DoorStateClassifier()
    
    try:
        # Load training data from CSV
        print(f"\n{'='*60}")
        print("Loading Training Data")
        print(f"{'='*60}")
        X, y = classifier.load_training_data_from_csv(csv_filename)
        
        # Preprocess data
        print(f"\n{'='*60}")
        print("Preprocessing Data")
        print(f"{'='*60}")
        X_scaled, y = classifier.preprocess_data(X, y)
        
        # Train model
        print(f"\n{'='*60}")
        print("Training SVM Model")
        print(f"{'='*60}")
        test_accuracy, best_params = classifier.train_model(X_scaled, y)
        
        # Save model
        classifier.save_model()
        
        print(f"\n{'='*70}")
        print(f"✓ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Final test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"Best parameters: {best_params}")
        print(f"Saved files:")
        print(f"  - door_classifier_2class.pkl")
        print(f"  - door_scaler_2class.pkl")
        print(f"  - confusion_matrix_2class.png")
        print(f"{'='*70}\n")
        
        # Feature importance (if linear kernel)
        if best_params.get('kernel') == 'linear':
            classifier.get_feature_importance()
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ ERROR DURING TRAINING")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()