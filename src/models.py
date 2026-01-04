from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Input, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class CropClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class YieldForecaster:
    def __init__(self, input_shape, model_type='LSTM'):
        """
        :param input_shape: (seq_length, n_features)
        :param model_type: 'LSTM' or 'GRU'
        """
        self.model = Sequential()
        if model_type == 'LSTM':
            self.model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
        else:
            self.model.add(GRU(64, input_shape=input_shape, return_sequences=False))
            
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1)) # Regression output
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
    def train(self, X_train, y_train, epochs=20, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
    def predict(self, X):
        return self.model.predict(X)

class RainfallTransformer:
    def __init__(self, input_shape, num_heads=4, ff_dim=64):
        """
        Transformer-based Time Series Predictor
        """
        inputs = Input(shape=input_shape)
        
        # Transformer Block
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=input_shape[1], num_heads=num_heads, dropout=0.1)(x, x)
        x = Dropout(0.1)(x)
        res = x + inputs # Residual connection
        
        # Feed Forward
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(input_shape[1])(x)
        x = x + res
        
        # Output
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
    def train(self, X_train, y_train, epochs=20, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
    def predict(self, X):
        return self.model.predict(X)
