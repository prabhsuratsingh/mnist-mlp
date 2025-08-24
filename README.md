# Two-Layer MLP on MNIST (from Scratch)

This project implements a **two-layer Multi-Layer Perceptron (MLP)** trained on the **MNIST dataset**.  
The goal is to demonstrate how a simple neural network can be built **from scratch** (without deep learning frameworks like TensorFlow or PyTorch) to classify handwritten digits (0–9).  

---

## ✨ Features
- Implementation of a **two-layer fully connected neural network**  
- **Forward propagation**, **loss computation**, and **backpropagation** implemented manually  
- Uses **stochastic gradient descent (SGD)** for optimization  
- Trained on the **MNIST handwritten digit dataset**  
- Achieves reasonable accuracy with minimal code  

---

## 📂 Project Structure
```
├── data/    
|   |── plots/      # stores all plots created from raw data
|   |── dataset.py              
|   |── preprocess.py              
├── models/
│   ├── mlp.py           
│   ├── helpers.py  
├── results/        # stores all graphs and images from training, validation and test
├── training/
│   ├── batch.py  
│   ├── loss.py  
│   ├── training_loop.py
├── training/
│   ├── metric_plots.py
│   ├── metrics.py
│   ├── plot_images.py
├── README.md              
├── .gitignore              
├── main.py         # entry point of MLP              
└── requirements.txt       
```


---

## 🧠 Model Architecture
The network consists of:
1. **Input Layer**: 784 units (28x28 flattened images)  
2. **Hidden Layer**: 50 neurons with Sigmoid activation  
3. **Output Layer**: 10 neurons with Sigmoid activation  

**Loss Function**: Sum of Squares  
**Optimizer**: SGD (with learning rate tuning)  

---

## 🚀 Installation & Usage
### 1. Clone Repository
```bash
git clone https://github.com/prabhsuratsingh/mnist-mlp.git
cd mnist-mlp
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Training
```bash
python main.py
```
