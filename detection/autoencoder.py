
from typing import Any
from detection.detector_base import BaseDetector
import keras
from keras import layers, models

class AutoencoderDetector(BaseDetector):
    """
    Docstring for AutoencoderDetector
    Autoencoder là một loại mạng nơ-ron được sử dụng để học biểu diễn (representation learning) của dữ liệu.
    Nó bao gồm hai phần chính: Encoder và Decoder.
    - Encoder: Phần này nhận dữ liệu đầu vào và nén nó thành một biểu diễn có kích thước nhỏ hơn, thường được gọi là "bottleneck". Mục tiêu của encoder là học cách nén dữ liệu một cách hiệu quả mà vẫn giữ được thông tin quan trọng.
    - Decoder: Phần này nhận biểu diễn nén từ encoder và cố gắng tái tạo lại dữ liệu gốc. Mục tiêu của decoder là học cách giải nén biểu diễn nén để tái tạo dữ liệu ban đầu.
    Autoencoder thường được sử dụng trong các bài toán giảm chiều dữ liệu, phát hiện bất thường (anomaly detection), và tạo dữ liệu mới (data generation). Trong bài toán phát hiện bất thường, autoencoder được huấn luyện trên dữ liệu bình thường, và sau đó sử dụng để tính toán lỗi tái tạo (reconstruction error) cho các điểm dữ liệu mới. Nếu lỗi tái tạo vượt quá một ngưỡng nhất định, điểm dữ liệu đó có thể được coi là bất thường.
    """
    def __init__(self, **kwargs):
        pass
        # self.model = model


    def build(self, input_shape=(30,), **kwargs) -> keras.Model:
        input_dim = input_shape[0]

        input_layer = layers.Input(shape=(input_dim,))
    
        # --- Thêm Noise (Để đối phó với 1% anomaly trong tập train) ---
        # GaussianNoise giúp mô hình không học vẹt các điểm dữ liệu cá biệt
        # noisy_input = layers.GaussianNoise(0.1)(input_layer)

        # --- Encoder ---
        # encoded = layers.Dense(14, activation='relu')(noisy_input)
        encoded = layers.Dense(21, activation='relu')(input_layer)
        bottleneck = layers.Dense(14, activation='relu')(encoded) # Không gian nén

        # --- Decoder ---
        decoded = layers.Dense(21, activation='relu')(bottleneck)
        output_layer = layers.Dense(input_dim, activation='linear')(decoded)

        # Kết nối thành Model
        autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def fit(self, X, epochs=50, batch_size=32):
        self.model = self.build()
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def score(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        reconstructed = self.model.predict(X)
        mse = ((X - reconstructed) ** 2).mean(axis=1)
        return mse
    
    
    def predict(self, X, threshold=None):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        reconstructed = self.model.predict(X)
        mse = ((X - reconstructed) ** 2).mean(axis=1)
        
        if threshold is None:
            # threshold = mse.mean() + 3 * mse.std()
            threshold = mse.mean() + 3 * mse.std()  # Điều chỉnh ngưỡng cho phù hợp với dữ liệu         
        return mse > threshold  # Return True for anomalies
    

    def save_model(self, file_path: str) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.save(file_path)

    def load_model(self, file_path: str) -> keras.Model:
        # We explicitly tell Pylance 'model' is a keras.Model
        model: keras.Model = keras.models.load_model(file_path) 
        self.model = model
        return self.model