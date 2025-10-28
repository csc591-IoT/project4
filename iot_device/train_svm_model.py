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

# ==============================
# SVM Model Training Script
# ==============================

class DoorStateClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_training_data(self, filename="training_data.json"):
        """Load training data from JSON file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Training data file {filename} not found")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        features = []
        labels = []
        
        for sample in data:
            features.append(sample['features'])
            labels.append(sample['label'])
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Loaded training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
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
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
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
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
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
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Closed', 'Open']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Closed', 'Open'], 
                   yticklabels=['Closed', 'Open'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy
    
    def save_model(self, model_filename="door_classifier_model.pkl", 
                   scaler_filename="door_scaler.pkl"):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.scaler, scaler_filename)
        
        print(f"Model saved to {model_filename}")
        print(f"Scaler saved to {scaler_filename}")
    
    def load_model(self, model_filename="door_classifier_model.pkl", 
                   scaler_filename="door_scaler.pkl"):
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
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance (for linear kernel)"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
            return importance
        else:
            print("Feature importance not available for non-linear kernels")
            return None

def analyze_training_data(filename="training_data.json"):
    """Analyze the training data"""
    if not os.path.exists(filename):
        print(f"Training data file {filename} not found")
        return
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    features = np.array([sample['features'] for sample in data])
    labels = np.array([sample['label'] for sample in data])
    
    print(f"Training Data Analysis")
    print(f"======================")
    print(f"Total samples: {len(data)}")
    print(f"Feature dimensions: {features.shape[1]}")
    print(f"Class distribution: {np.bincount(labels)}")
    print(f"Closed samples: {np.sum(labels == 0)}")
    print(f"Open samples: {np.sum(labels == 1)}")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"Mean: {np.mean(features, axis=0)[:5]}...")  # Show first 5
    print(f"Std: {np.std(features, axis=0)[:5]}...")    # Show first 5
    print(f"Min: {np.min(features, axis=0)[:5]}...")    # Show first 5
    print(f"Max: {np.max(features, axis=0)[:5]}...")   # Show first 5
    
    # Check for any invalid values
    invalid_count = np.sum(~np.isfinite(features))
    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid values found in features")

def main():
    print("Door State SVM Classifier Training")
    print("==================================")
    
    # Analyze training data
    analyze_training_data()
    
    # Initialize classifier
    classifier = DoorStateClassifier()
    
    try:
        # Load training data
        X, y = classifier.load_training_data()
        
        # Preprocess data
        X_scaled, y = classifier.preprocess_data(X, y)
        
        # Train model
        print("\nTraining SVM model...")
        test_accuracy = classifier.train_model(X_scaled, y)
        
        # Save model
        classifier.save_model()
        
        print(f"\nTraining completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.3f}")
        
        # Feature importance (if linear kernel)
        importance = classifier.get_feature_importance()
        if importance is not None:
            print(f"\nTop 10 most important features:")
            top_indices = np.argsort(importance)[-10:]
            for i, idx in enumerate(reversed(top_indices)):
                print(f"{i+1}. Feature {idx}: {importance[idx]:.3f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have collected training data first using training_data_collector.py")

if __name__ == "__main__":
    main()
