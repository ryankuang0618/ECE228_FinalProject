import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers # Import regularizers
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FitRecSpeedPredictor:
    def __init__(self, window_duration=30, prediction_duration=30):
        """
        Initialize the speed prediction model

        Args:
            window_duration (int): Duration in seconds of previous time window for prediction
            prediction_duration (int): Duration in seconds of future time window to predict
        """
        self.window_duration = window_duration
        self.prediction_duration = prediction_duration
        self.model = None
        self.speed_scaler = MinMaxScaler()
        self.hr_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.sampling_rate = 1  # Will be determined from data

    def load_endomondo_data(self):
        """
        Load and preprocess Endomondo HR dataset using the specified path
        """
        try:
            path = Path("data/")
            out_path = str(path / "run_sample_10000.npy") # Assuming this path exists and contains the data
            data = np.load(out_path, allow_pickle=True) # Use allow_pickle=True if data is stored as python objects

            print(f"Loaded data type: {type(data)}")

            # Process the data based on the example structure provided
            if isinstance(data, dict):
                return self.process_endomondo_dict(data)
            elif isinstance(data, np.ndarray) and data.ndim == 0: # Handle 0-dim array containing a dict/list
                unpacked_data = data.item() # Unpack the object
                if isinstance(unpacked_data, dict):
                    return self.process_endomondo_dict(unpacked_data)
                elif isinstance(unpacked_data, list):
                    all_sessions = []
                    for i, session in enumerate(unpacked_data):
                        if isinstance(session, dict):
                            session_df = self.process_endomondo_dict(session, session_id=i)
                            if session_df is not None and len(session_df) > 0:
                                all_sessions.append(session_df)
                        if i >= 50: # Limit sessions for memory efficiency
                            break
                    if all_sessions:
                        combined_df = pd.concat(all_sessions, ignore_index=True)
                        print(f"Combined {len(all_sessions)} sessions with {len(combined_df)} total data points")
                        return combined_df
                    else:
                        print("No valid sessions found from unpacked data")
                        return self.generate_sample_data()
                else:
                    print(f"Unknown unpacked data format: {type(unpacked_data)}")
                    return self.generate_sample_data()
            elif isinstance(data, list):
                # Process multiple sessions
                all_sessions = []
                for i, session in enumerate(data):
                    if isinstance(session, dict):
                        session_df = self.process_endomondo_dict(session, session_id=i)
                        if session_df is not None and len(session_df) > 0:
                            all_sessions.append(session_df)
                    # Limit sessions for memory efficiency
                    if i >= 50:  # Process maximum 50 sessions
                        break

                if all_sessions:
                    combined_df = pd.concat(all_sessions, ignore_index=True)
                    print(f"Combined {len(all_sessions)} sessions with {len(combined_df)} total data points")
                    return combined_df
                else:
                    print("No valid sessions found")
                    return self.generate_sample_data()
            else:
                print(f"Unknown data format: {type(data)}")
                return self.generate_sample_data()

        except Exception as e:
            print(f"Error loading Endomondo data: {e}")
            print("Generating sample data for demonstration...")
            return self.generate_sample_data()

    def process_endomondo_dict(self, data, session_id=None):
        """
        Process single Endomondo session dictionary
        """
        try:
            # Extract the required fields
            required_fields = ['tar_heart_rate', 'tar_derived_speed', 'timestamp']
            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                print(f"Missing required fields: {missing_fields} in session {session_id}")
                print(f"Available fields: {list(data.keys())}")
                return None

            # Extract data
            heart_rate = np.array(data['tar_heart_rate'])
            speed = np.array(data['tar_derived_speed'])
            timestamp = np.array(data['timestamp'])

            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamp,
                'heart_rate': heart_rate,
                'speed': speed
            })

            if session_id is not None:
                df['session_id'] = session_id

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            print(f"Session {session_id} data shape: {df.shape}")
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min())/60:.1f} minutes")

            return self.clean_endomondo_data(df)

        except Exception as e:
            print(f"Error processing session {session_id}: {e}")
            return None

    def clean_endomondo_data(self, df):
        """
        Clean and preprocess Endomondo data
        """
        print(f"Raw data shape: {df.shape}")

        # Remove invalid values
        df = df.dropna()
        df = df[(df['speed'] >= 0) & (df['heart_rate'] > 0)]

        # Calculate time differences to determine sampling rate
        if len(df) > 1:
            time_diffs = np.diff(df['timestamp'])
            median_interval = np.median(time_diffs)
            if median_interval > 0: # Ensure sampling rate is positive
                self.sampling_rate = median_interval
            else:
                self.sampling_rate = 1 # Default if no valid diffs
        else:
            self.sampling_rate = 1 # Default for very small dataframes

        print(f"Median sampling interval: {self.sampling_rate:.2f} seconds")

        # Remove extreme outliers using IQR method
        for col in ['speed', 'heart_rate']:
            if len(df) > 0: # Check if DataFrame is not empty
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.0 * IQR  # More lenient for real data
                upper_bound = Q3 + 2.0 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        print(f"Cleaned data shape: {df.shape}")
        if len(df) > 0:
            print(f"Speed range: {df['speed'].min():.2f} - {df['speed'].max():.2f}")
            print(f"Heart rate range: {df['heart_rate'].min():.0f} - {df['heart_rate'].max():.0f}")
        else:
            print("No data left after cleaning.")

        return df

    def generate_sample_data(self, n_samples=5000):
        """
        Generate sample fitness data for demonstration
        """
        print("Generating sample data for demonstration...")
        np.random.seed(42)

        # Simulate realistic fitness data
        time = np.arange(n_samples)

        # Base patterns
        base_speed = 8 + 3 * np.sin(time * 0.01) + np.random.normal(0, 1, n_samples)
        base_hr = 140 + 20 * np.sin(time * 0.01 + 0.5) + np.random.normal(0, 5, n_samples)

        # Add correlation between speed and heart rate
        correlation_factor = 0.3
        hr_adjustment = correlation_factor * (base_speed - np.mean(base_speed))
        heart_rate = base_hr + hr_adjustment

        # Ensure positive values
        speed = np.maximum(base_speed, 0.1)
        heart_rate = np.maximum(heart_rate, 60)

        df = pd.DataFrame({
            'timestamp': np.arange(n_samples), # Add timestamp for consistency
            'speed': speed,
            'heart_rate': heart_rate
        })

        return df

    def create_time_based_sequences(self, data):
        """
        Create time-based sequences for prediction

        Args:
            data (pd.DataFrame): Data with timestamp, speed, and heart_rate columns
        """
        print(f"Creating time-based sequences...")
        print(f"Window duration: {self.window_duration}s, Prediction duration: {self.prediction_duration}s")

        # Calculate number of samples needed for each window
        window_samples = int(self.window_duration / self.sampling_rate)
        prediction_samples = int(self.prediction_duration / self.sampling_rate)

        if window_samples == 0:
            raise ValueError(f"window_samples is 0. Check window_duration ({self.window_duration}) and sampling_rate ({self.sampling_rate}).")
        if prediction_samples == 0:
            raise ValueError(f"prediction_samples is 0. Check prediction_duration ({self.prediction_duration}) and sampling_rate ({self.sampling_rate}).")


        print(f"Window samples: {window_samples}, Prediction samples: {prediction_samples}")

        # Prepare features (using 'speed' and 'heart_rate' which correspond to 'tar_derived_speed' and 'tar_heart_rate')
        speed_scaled = self.speed_scaler.fit_transform(data[['speed']])
        hr_scaled = self.hr_scaler.fit_transform(data[['heart_rate']])
        features = np.concatenate([speed_scaled, hr_scaled], axis=1)

        X, y = [], []
        timestamps_x, timestamps_y = [], []

        # Create overlapping windows with 50% overlap for more training data
        step_size = max(1, window_samples // 2)

        # Ensure there's enough data for at least one full sequence
        if len(data) < window_samples + prediction_samples:
            print(f"Not enough data to create sequences. Minimum required: {window_samples + prediction_samples} samples. Found: {len(data)} samples.")
            return np.array([]), np.array([]), np.array([]), np.array([])


        for i in range(0, len(data) - window_samples - prediction_samples + 1, step_size):
            # Input window
            x_window = features[i:i + window_samples]
            x_times = data['timestamp'].iloc[i:i + window_samples]

            # Future window for prediction target
            future_start = i + window_samples
            future_end = future_start + prediction_samples
            future_speeds = data['speed'].iloc[future_start:future_end]
            future_times = data['timestamp'].iloc[future_start:future_end]

            # Calculate average speed in the future window
            avg_future_speed = future_speeds.mean()

            X.append(x_window)
            y.append(avg_future_speed)
            timestamps_x.append(x_times.values)
            timestamps_y.append(future_times.values)

        X = np.array(X)
        y = np.array(y)

        # Scale target values
        if len(y) > 0: # Ensure y is not empty before scaling
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = np.array([]) # Return empty if no sequences were created

        print(f"Created {len(X)} sequences")
        print(f"Input shape: {X.shape}, Target shape: {y_scaled.shape}")

        return X, y_scaled, np.array(timestamps_x), np.array(timestamps_y)

    def create_model(self, input_shape):
        """
        Create LSTM model for speed prediction
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=regularizers.l2(0.001), # Added L2 regularization
                 recurrent_regularizer=regularizers.l2(0.001)), # Added L2 regularization
            Dropout(0.3), # Increased dropout rate
            BatchNormalization(),

            LSTM(64, return_sequences=True,
                 kernel_regularizer=regularizers.l2(0.001), # Added L2 regularization
                 recurrent_regularizer=regularizers.l2(0.001)), # Added L2 regularization
            Dropout(0.3), # Increased dropout rate
            BatchNormalization(),

            LSTM(32, return_sequences=False,
                 kernel_regularizer=regularizers.l2(0.001), # Added L2 regularization
                 recurrent_regularizer=regularizers.l2(0.001)), # Added L2 regularization
            Dropout(0.3), # Increased dropout rate

            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # Added L2 regularization
            Dropout(0.2), # Increased dropout rate for dense layers too
            Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # Added L2 regularization
            Dense(1)  # Predict single average speed value
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def split_data(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into train, validation, and test sets chronologically
        
        Args:
            data (pd.DataFrame): Input data with timestamp, speed, heart_rate columns
            train_ratio (float): Proportion for training data
            val_ratio (float): Proportion for validation data  
            test_ratio (float): Proportion for test data
        
        Returns:
            tuple: (train_data, val_data, test_data, split_info)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        print("Creating time-based sequences for data splitting...")
        X, y, timestamps_x, timestamps_y = self.create_time_based_sequences(data)
        
        if len(X) == 0:
            print("No sequences created. Cannot split data.")
            return None, None, None, None
        
        # Calculate split indices chronologically
        n_total = len(X)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        # Split the data chronologically
        X_train = X[:train_end]
        y_train = y[:train_end]
        timestamps_train = timestamps_y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        timestamps_val = timestamps_y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        timestamps_test = timestamps_y[val_end:]
        
        # Store split data for later use
        self.split_data_cache = {
            'X_train': X_train, 'y_train': y_train, 'timestamps_train': timestamps_train,
            'X_val': X_val, 'y_val': y_val, 'timestamps_val': timestamps_val,
            'X_test': X_test, 'y_test': y_test, 'timestamps_test': timestamps_test
        }
        
        split_info = {
            'total_sequences': n_total,
            'train_sequences': len(X_train),
            'val_sequences': len(X_val), 
            'test_sequences': len(X_test),
            'train_ratio': len(X_train) / n_total,
            'val_ratio': len(X_val) / n_total,
            'test_ratio': len(X_test) / n_total
        }
        
        print(f"\nData Split Summary:")
        print(f"Total sequences: {split_info['total_sequences']}")
        print(f"Training: {split_info['train_sequences']} sequences ({split_info['train_ratio']:.1%})")
        print(f"Validation: {split_info['val_sequences']} sequences ({split_info['val_ratio']:.1%})")
        print(f"Test: {split_info['test_sequences']} sequences ({split_info['test_ratio']:.1%})")
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return (X_train, y_train, timestamps_train), (X_val, y_val, timestamps_val), (X_test, y_test, timestamps_test), split_info

    def train_model_only(self, train_data, val_data, epochs=100, batch_size=32, verbose=1):
        """
        Train the model using only train and validation data (no test evaluation)
        
        Args:
            train_data (tuple): (X_train, y_train, timestamps_train)
            val_data (tuple): (X_val, y_val, timestamps_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level for training
            
        Returns:
            keras.callbacks.History: Training history
        """
        X_train, y_train, _ = train_data
        X_val, y_val, _ = val_data
        
        if len(X_train) == 0:
            print("No training data available.")
            return None
            
        # Create and compile model
        self.model = self.create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        if verbose > 0:
            print(self.model.summary())
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=10,
            min_lr=1e-7
        )
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return history

    def evaluate_model(self, test_data, plot_results=True):
        """
        Evaluate the trained model on test data
        
        Args:
            test_data (tuple): (X_test, y_test, timestamps_test)
            plot_results (bool): Whether to plot evaluation results
            
        Returns:
            dict: Evaluation results including metrics and predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train_model_only() first.")
            
        X_test, y_test, timestamps_test = test_data
        
        if len(X_test) == 0:
            print("No test data available for evaluation.")
            return {
                'mse': np.nan, 'mae': np.nan, 'rmse': np.nan,
                'y_test': np.array([]), 'y_pred': np.array([]),
                'timestamps': np.array([])
            }
        
        print("\nEvaluating model on test set...")
        
        # Evaluate model
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Inverse transform for interpretable metrics
        y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_orig = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / np.maximum(y_test_orig, 1e-8))) * 100
        r2 = 1 - (np.sum((y_test_orig - y_pred_orig) ** 2) / np.sum((y_test_orig - np.mean(y_test_orig)) ** 2))
        
        results = {
            'test_loss': test_loss,
            'test_mae_scaled': test_mae,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'y_test': y_test_orig,
            'y_pred': y_pred_orig,
            'timestamps': timestamps_test,
            'mean_actual': y_test_orig.mean(),
            'mean_predicted': y_pred_orig.mean(),
            'std_actual': y_test_orig.std(),
            'std_predicted': y_pred_orig.std()
        }
        
        print(f"\nTest Results:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean actual speed: {y_test_orig.mean():.2f} ± {y_test_orig.std():.2f}")
        print(f"Mean predicted speed: {y_pred_orig.mean():.2f} ± {y_pred_orig.std():.2f}")
        
        if plot_results:
            self.plot_predictions(y_test_orig, y_pred_orig)
            
        return results

    def save_test_data(self, test_data, filepath="test_data.npz"):
        """
        Save test data to file for later evaluation
        
        Args:
            test_data (tuple): (X_test, y_test, timestamps_test)
            filepath (str): Path to save the test data
        """
        X_test, y_test, timestamps_test = test_data
        
        np.savez(filepath, 
                 X_test=X_test, 
                 y_test=y_test, 
                 timestamps_test=timestamps_test)
        print(f"Test data saved to {filepath}")

    def load_test_data(self, filepath="test_data.npz"):
        """
        Load test data from file
        
        Args:
            filepath (str): Path to load the test data from
            
        Returns:
            tuple: (X_test, y_test, timestamps_test)
        """
        data = np.load(filepath)
        X_test = data['X_test']
        y_test = data['y_test'] 
        timestamps_test = data['timestamps_test']
        print(f"Test data loaded from {filepath}")
        return X_test, y_test, timestamps_test

    def train(self, data, test_size=0.2, validation_size=0.1, epochs=100, batch_size=32):
        """
        Train the speed prediction model (legacy method - updated to use new splitting approach)

        Args:
            data (pd.DataFrame): Training data with timestamp, speed, heart_rate columns
            test_size (float): Proportion of data for testing
            validation_size (float): Proportion of training data for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        # Calculate ratios for new splitting method
        train_ratio = 1 - test_size
        val_ratio = train_ratio * validation_size
        train_ratio = train_ratio * (1 - validation_size)
        
        # Use new splitting method
        train_data, val_data, test_data, split_info = self.split_data(
            data, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_size
        )
        
        if train_data is None:
            return None, None
            
        # Train model
        history = self.train_model_only(train_data, val_data, epochs, batch_size)
        
        if history is None:
            return None, None
            
        # Evaluate on test set
        results = self.evaluate_model(test_data, plot_results=True)
        
        # Plot training history
        self.plot_training_history(history)
        
        return history, results

    def predict(self, input_sequence):
        """
        Make prediction for new input sequence

        Args:
            input_sequence (np.array): Input sequence of shape (window_samples, 2)
                                     where columns are [speed, heart_rate]
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Calculate expected window size
        window_samples = int(self.window_duration / self.sampling_rate)

        # Ensure correct shape
        if input_sequence.shape[0] != window_samples or input_sequence.shape[1] != 2:
            raise ValueError(f"Input should have shape ({window_samples}, 2), got {input_sequence.shape}")

        # Scale input
        speed_scaled = self.speed_scaler.transform(input_sequence[:, [0]])
        hr_scaled = self.hr_scaler.transform(input_sequence[:, [1]])
        input_scaled = np.concatenate([speed_scaled, hr_scaled], axis=1)

        # Reshape for prediction
        input_scaled = input_scaled.reshape(1, window_samples, 2)

        # Make prediction
        pred_scaled = self.model.predict(input_scaled, verbose=0)

        # Inverse transform
        pred_orig = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        return pred_orig[0, 0]

    def predict_from_dataframe(self, df, start_idx):
        """
        Make prediction using data from DataFrame starting at given index

        Args:
            df (pd.DataFrame): DataFrame with speed and heart_rate columns
            start_idx (int): Starting index for the input window
        """
        window_samples = int(self.window_duration / self.sampling_rate)

        if start_idx + window_samples > len(df):
            raise ValueError("Not enough data for prediction window")

        # Use 'speed' and 'heart_rate' columns (which map to tar_derived_speed and tar_heart_rate)
        input_data = df[['speed', 'heart_rate']].iloc[start_idx:start_idx + window_samples].values
        return self.predict(input_data)

    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # MAE
        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, y_true, y_pred, n_samples=200):
        """
        Plot prediction results
        """
        # Limit samples for clearer visualization
        if len(y_true) > n_samples:
            indices = np.linspace(0, len(y_true)-1, n_samples, dtype=int)
            y_true_plot = y_true[indices]
            y_pred_plot = y_pred[indices]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred

        plt.figure(figsize=(15, 8))

        # Prediction vs actual
        plt.subplot(2, 1, 1)
        plt.plot(y_true_plot, label='Actual Speed', alpha=0.7)
        plt.plot(y_pred_plot, label='Predicted Speed', alpha=0.7)
        plt.title('Speed Prediction Results')
        plt.xlabel('Time')
        plt.ylabel('Speed')
        plt.legend()
        plt.grid(True)

        # Scatter plot
        plt.subplot(2, 1, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Speed')
        plt.ylabel('Predicted Speed')
        plt.title('Predicted vs Actual Speed')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Example usage
def main():
    """
    Main function to demonstrate the speed prediction model with Endomondo data
    """
    # Initialize predictor
    predictor = FitRecSpeedPredictor(window_duration=30, prediction_duration=30) # Use window_duration and prediction_duration

    # Load Endomondo data using the specified method
    data = predictor.load_endomondo_data()

    if data is None or data.empty:
        print("No data loaded or data is empty. Exiting.")
        return None, None

    print("\nData sample:")
    print(data.head())
    print(f"\nData statistics:")
    print(data.describe())

    # Train model
    history, results = predictor.train(
        data,
        epochs=50,  # Reduce for demonstration
        batch_size=64
    )

    if results is None:
        print("Model training failed or no results were generated.")
        return predictor, None

    # Example prediction
    print("\nMaking sample prediction...")
    # Ensure there's enough data for a sample prediction
    window_samples = int(predictor.window_duration / predictor.sampling_rate)
    if len(data) >= window_samples:
        sample_input = data[['speed', 'heart_rate']].iloc[-window_samples:].values
        prediction = predictor.predict(sample_input)
        print(f"Predicted next average speed for {predictor.prediction_duration}s: {prediction:.2f}")

        # To get the actual future speed for comparison, we need to manually calculate it
        # This will depend on the last actual future window used in the test set,
        # or calculate it directly from the raw data.
        future_start_idx = len(data) - predictor.prediction_duration
        if future_start_idx >= 0:
            actual_future_speed = data['speed'].iloc[future_start_idx:].mean()
            print(f"Actual average speed in the last {predictor.prediction_duration}s: {actual_future_speed:.2f}")
        else:
            print("Not enough data to calculate actual average speed for comparison.")
    else:
        print(f"Not enough data to make a sample prediction. Need at least {window_samples} samples.")

    return predictor, results

def demonstrate_data_splitting():
    """
    Comprehensive example of data splitting and separate train/test workflow
    """
    print("=== Demonstrating Data Splitting and Evaluation Workflow ===\n")
    
    # Initialize predictor
    predictor = FitRecSpeedPredictor(window_duration=30, prediction_duration=30)
    
    # Load data
    print("1. Loading data...")
    data = predictor.load_endomondo_data()
    
    if data is None or data.empty:
        print("No data loaded. Exiting.")
        return None, None, None, None
    
    print(f"Loaded data shape: {data.shape}")
    
    # Split data into train/val/test
    print("\n2. Splitting data chronologically...")
    train_data, val_data, test_data, split_info = predictor.split_data(
        data, 
        train_ratio=0.7,   # 70% for training
        val_ratio=0.15,    # 15% for validation  
        test_ratio=0.15    # 15% for testing
    )
    
    if train_data is None:
        print("Data splitting failed. Exiting.")
        return None, None, None, None
    
    # Save test data for later evaluation
    print("\n3. Saving test data for later evaluation...")
    predictor.save_test_data(test_data, "heart_rate_test_data.npz")
    
    # Train model (without test evaluation)
    print("\n4. Training model...")
    history = predictor.train_model_only(
        train_data, 
        val_data, 
        epochs=50,  # Reduced for demonstration
        batch_size=32,
        verbose=1
    )
    
    if history is None:
        print("Training failed. Exiting.")
        return None, None, None, None
    
    # Plot training history
    print("\n5. Plotting training history...")
    predictor.plot_training_history(history)
    
    # Now evaluate on test set
    print("\n6. Evaluating trained model on test set...")
    test_results = predictor.evaluate_model(test_data, plot_results=True)
    
    # Demonstrate loading test data from file and re-evaluating
    print("\n7. Demonstrating loading test data from file...")
    loaded_test_data = predictor.load_test_data("heart_rate_test_data.npz")
    
    print("\n8. Re-evaluating with loaded test data...")
    test_results_reloaded = predictor.evaluate_model(loaded_test_data, plot_results=False)
    
    # Compare results
    print("\n9. Comparing evaluation results:")
    print(f"Original test MSE: {test_results['mse']:.4f}")
    print(f"Reloaded test MSE: {test_results_reloaded['mse']:.4f}")
    print(f"Results match: {np.isclose(test_results['mse'], test_results_reloaded['mse'])}")
    
    return predictor, history, test_results, split_info

def load_and_train_endomondo():
    """
    Simplified function to load and train with Endomondo data
    """
    # Initialize and train predictor
    predictor = FitRecSpeedPredictor(window_duration=30, prediction_duration=30) # Use window_duration and prediction_duration

    # Load and process data
    processed_data = predictor.load_endomondo_data()

    if processed_data is None or processed_data.empty:
        print("No processed data available to train the model.")
        return None, None, None

    # Train the model
    history, results = predictor.train(processed_data, epochs=100, batch_size=32)

    return predictor, results, processed_data

if __name__ == "__main__":
    predictor, results = main()