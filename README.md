# AI Casting Guard: Automated Casting Defect Classification using Multiple CNN

Developed a deep learning system using multiple CNN models to detect and classify casting defects from industrial images. Implemented preprocessing, training, and evaluation to improve accuracy, enabling efficient and reliable automated quality inspection in manufacturing.

AI Casting Guard is a high-performance system designed for real-time industrial quality control. It compares results across multiple state-of-the-art architectures.

## 🚀 Features
- **Multi-Model Analysis**: Compares results across three optimized architectures:
  - **Custom CNN**: Features VGG-style blocks, Squeeze-and-Excitation (SE) attention, and SiLU activations.
  - **MobileNetV2**: Optimized for speed and edge deployment with selective fine-tuning.
  - **ResNet50**: High-accuracy residual network for complex feature extraction.
- **Web-Based Interface**: A modern, responsive UI for easy image uploads and instant analysis results.
- **Optimized Training Pipeline**: Includes cosine-decay learning rate schedules, warmup epochs, and data augmentation tailored for industrial inspection.
- **Real-Time Inference**: Powered by a Flask backend for rapid processing.

## 📊 Dataset
The models are trained on the **Real-life Industrial Dataset of Casting Product**, which contains top-view images of submersible pump impellers.

- **Dataset Link**: [Kaggle - Casting Product Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
- **Classes**: Blowhole, Crack, Flash, and Normal (Non-defective).

## 🛠️ Tech Stack
- **Core**: Python 3.10+
- **Deep Learning**: TensorFlow, Keras 3
- **Web Backend**: Flask, Flask-CORS
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Libraries**: NumPy, Matplotlib, Scikit-learn, Pillow, OpenCV

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tamilselvan526/Automated-casting-defect-classification-using-multiple-cnn.git
   cd Automated-casting-defect-classification-using-multiple-cnn
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Models**:
   ```bash
   python main.py
   ```

5. **Run the Web Server**:
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`.

## 📁 Project Structure
- `main.py`: The core training script for all three models.
- `predict.py`: CLI-based prediction tool.
- `app.py`: Flask server for the web application.
- `static/`: Frontend assets (UI design, logic).
- `model/`: Directory where trained `.keras` files and metadata are stored.
- `dataset/`: (Ignored by Git) Should contain the images following this structure:
  ```text
  dataset/
  ├── train/
  │   ├── blowhole/
  │   ├── crack/
  │   ├── flash/
  │   └── normal/
  └── test/
      ├── blowhole/
      ├── crack/
      ├── flash/
      └── normal/
  ```

## 📄 License
This project is for educational and research purposes in the field of industrial AI.
