# nn-addition-app

An interactive demo of a simple neural network

## 🧠 Neural Network for Learning Addition

This is a simple Streamlit app that trains a small neural network to **learn how to add two numbers** between 0 and 20. It visualizes the model's performance and allows users to test predictions interactively.

---

## 📊 Features

- A neural network built from scratch using NumPy
- Trains on a minimal dataset of addition pairs
- Shows a **heatmap** of correct vs. incorrect predictions across all (a, b) pairs
- Lets you test predictions using interactive sliders
- Clean UI built with [Streamlit](https://streamlit.io)

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nn-addition-app.git
cd nn-addition-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run nn_addition_streamlit.py
```

The app will open in your browser at http://localhost:8501

## 🌍 Try It Online

You can use the app live on Streamlit Cloud:

[Launch the app](https://your-app-url-here)

## 📂 File Overview

| File | Description |
| ---- | ----------- |
| nn_addition_streamlit.py | Main Streamlit app |
| requirements.txt | Dependencies needed to run the app |

## 💡 Ideas for Extension

- Let users add/remove training pairs
- Visualize the internal computation of the network
- Add options for different activation functions
- Switch to regression (using linear output) instead of classification

## 🧑‍💻 Author

Made with ❤️ by me

## 📜 License

This project is licensed under the MIT License.
