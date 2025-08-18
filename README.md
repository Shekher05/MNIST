# 📝 MNIST Handwritten Digit Recognition  

This project builds and trains a **Neural Network** to recognize handwritten digits (0–9) using the **MNIST dataset**.  
The model is implemented with **TensorFlow & Keras** and demonstrates a **3-layer neural network** with **ReLU** and **Softmax** activations.  
It also includes **logging** and **exception handling** for better maintainability and debugging.  

---

## 🚀 Features  
- 📊 **Dataset:** MNIST (70,000 grayscale digit images, 28x28 pixels).  
- 🧠 **Model Architecture:**  
  - 3 Dense Layers.  
  - **ReLU** for hidden layers.  
  - **Softmax** for output classification (10 classes).  
- ⚙️ **Implementation Highlights:**  
  - Custom modules for **data loading, logging, exception handling, and plotting**.  
  - Model training with **Keras Dense layers**.  
  - Logging for tracking training performance.  
  - Exception handling for robust execution.  

---

## 📂 Project Structure  
```bash
MNIST/
│-- artifacts/              # Saved trained models
│   ├── mnist_model.h5
│   └── mnist_model.keras
│
│-- logs/                   # Training logs
│-- mnist_data/             # Dataset (downloaded/processed)
│
│-- src/                    # Core source code
│   ├── data.py             # Data loading & preprocessing
│   ├── exception_h.py      # Exception handling utilities
│   ├── forward_.py         # Forward propagation / training logic
│   ├── logger.py           # Logging configuration
│   ├── Model.py            # Model definition
│   ├── plot.py             # Visualization (accuracy/loss graphs etc.)
│
│-- .gitignore
│-- main.py                 # Main script (training entry point)
│-- requirement.txt         # Project dependencies
│-- README.md               # Project documentation

---

## ▶️ How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/Shekher05/MNIST.git
   cd MNIST-Digit-Recognition

2. **Install Dependencies:**
   ```bash
   pip install -r requirement.txt

3. **Run Training:**
   ```bash
   python main.py

---

## 🔮 Future Improvements  
- 🌐 Add a **web interface** (Flask/Django/Streamlit) for real-time digit recognition.  
- 📊 Implement a **confusion matrix** and more detailed evaluation metrics.  
- 📦 Save and load models more efficiently with versioning.  
- 🚀 Deploy the model as a **REST API** for wider use.  
- 🎨 Improve visualization (loss/accuracy curves, sample predictions).  

---

## 📬 Contact  
👤 **Shekher J Singh**  
📧 [10shekhersingh@gmail.com](mailto:10shekhersingh@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/shekher-j-singh-277912277/)  

---

⭐ If you found this project useful, don’t forget to **star the repo**!  



