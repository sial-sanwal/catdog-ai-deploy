# 🐱 CatDog Vision AI 🐶

A complete end-to-end Deep Learning project for classifying images as Cats or Dogs using Convolutional Neural Networks (CNN).

## 🎯 Features

- **CNN Model Training**: Train a custom CNN model on the Cats vs Dogs dataset
- **RESTful API**: Flask backend with `/api/predict` endpoint
- **Modern Frontend**: Beautiful, responsive web interface for image upload and prediction
- **Real-time Classification**: Upload images and get instant predictions with confidence scores

## 📋 Requirements

- Python 3.11+
- uv package manager
- Internet connection (for downloading dataset)

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
uv sync
```

This will install all required packages:
- TensorFlow/Keras
- Flask
- NumPy
- Pillow
- OpenCV

### 2. Train the Model

First, you need to train the CNN model on the dataset:

```bash
uv run python train_model.py
```

This script will:
- Automatically download the Cats vs Dogs dataset
- Train a CNN model with the following architecture:
  - 4 Convolutional layers with MaxPooling
  - Dropout for regularization
  - Dense layers for classification
- Save the trained model as `catdog_model.h5`
- Target accuracy: ~85% or higher

**Note**: Training may take 30-60 minutes depending on your hardware.

### 3. Start the Flask Server

Once the model is trained, start the Flask API server:

```bash
uv run python app.py
```

The server will start on `http://localhost:5000`

### 4. Open the Frontend

Open your web browser and navigate to:
```
http://localhost:5000
```

## 🐳 Docker Deployment

### Using Dockerfile

1. **Build the Docker image**:
   ```bash
   docker build -t catdog-vision-ai .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 5000:5000 --name catdog-ai catdog-vision-ai
   ```

3. **Access the application**:
   ```
   http://localhost:5000
   ```

### Using Docker Compose (Recommended)

1. **Start the application**:
   ```bash
   docker-compose up -d
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Stop the application**:
   ```bash
   docker-compose down
   ```

**Note**: Make sure you have trained the model (`catdog_model.h5`) before building the Docker image, as the model file is required for the application to run.

## 🎨 Usage

1. **Upload an Image**: 
   - Click the upload area or drag and drop an image
   - Supported formats: PNG, JPG, JPEG (max 10MB)

2. **Get Prediction**:
   - Click the "Predict" button
   - The model will analyze the image and return:
     - Predicted class: **Cat** 🐱 or **Dog** 🐶
     - Confidence score (percentage)

3. **Clear and Try Again**:
   - Click "Clear" to upload a new image

## 📁 Project Structure

```
catdog-ai-deploy/
├── app.py                  # Flask backend API
├── train_model.py          # Model training script
├── main.py                 # Main entry point
├── pyproject.toml          # Project dependencies
├── catdog_model.h5         # Trained model (generated after training)
├── Dockerfile              # Docker container configuration
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Files to exclude from Docker build
├── cats_and_dogs_filtered/ # Dataset (downloaded automatically)
└── static/
    ├── index.html          # Frontend HTML
    ├── style.css           # Styling
    └── script.js           # Frontend JavaScript
```

## 🧠 Model Architecture

The CNN model consists of:

1. **Convolutional Layers**:
   - Conv2D(32 filters) → MaxPooling2D
   - Conv2D(64 filters) → MaxPooling2D
   - Conv2D(128 filters) → MaxPooling2D
   - Conv2D(128 filters) → MaxPooling2D

2. **Dense Layers**:
   - Flatten
   - Dropout(0.5)
   - Dense(512, relu)
   - Dense(1, sigmoid) - Binary classification output

3. **Image Size**: 150x150 pixels
4. **Optimizer**: RMSprop with learning rate 0.001
5. **Loss Function**: Binary crossentropy

## 🔌 API Endpoints

### `GET /`
Serves the frontend HTML page.

### `POST /api/predict`
Upload an image for classification.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response**:
```json
{
    "class": "Cat",
    "confidence": 0.9456,
    "raw_confidence": 0.0544
}
```

### `GET /api/health`
Health check endpoint.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## 🛠️ Development

### Running Tests
You can test the API using curl:

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@path/to/your/image.jpg"
```

### Modifying the Model
Edit `train_model.py` to:
- Adjust image size (default: 150x150)
- Change number of epochs (default: 15)
- Modify CNN architecture
- Adjust batch size (default: 32)

## 📝 Notes

- The dataset is automatically downloaded from Google's ML Education dataset
- Model training includes data augmentation for better generalization
- The frontend uses vanilla JavaScript (no frameworks required)
- CORS is enabled for API access

## 👨‍💻 Developer

Developed by **Sanwal Khan** — AI Engineer

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset provided by Google's ML Education resources
- Built with TensorFlow/Keras and Flask

