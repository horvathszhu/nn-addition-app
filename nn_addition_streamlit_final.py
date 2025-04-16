# nn_addition_streamlit.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

ANNOTATION_FONT = 15

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def generate_minimal_training_data():
    selected_pairs = st.session_state.training_pairs
    X_train = [[i, j] for i, j in selected_pairs]
    y_train = [[i + j] for i, j in selected_pairs]
    return np.array(X_train) / 20, np.array(y_train) / 40

def initialize_weights(input_size, n_hidden, output_size):
    np.random.seed(42)
    W1 = np.random.uniform(-1, 1, (input_size, n_hidden))
    W2 = np.random.uniform(-1, 1, (n_hidden, output_size))
    b1 = np.zeros((1, n_hidden))
    b2 = np.zeros((1, output_size))
    return W1, W2, b1, b2

def train_model(X_train, y_train, n_hidden=5, learning_rate=0.1, n_epochs=10000):
    input_size = 2
    output_size = 1
    W1, W2, b1, b2 = initialize_weights(input_size, n_hidden, output_size)

    for epoch in range(n_epochs):
        z1 = np.dot(X_train, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        output = sigmoid(z2)

        error = y_train - output
        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative(a1)

        W2 += a1.T.dot(d_output) * learning_rate
        b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        W1 += X_train.T.dot(d_hidden) * learning_rate
        b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    return W1, W2, b1, b2

def predict(a, b, W1, W2, b1, b2):
    inputs = np.array([[a, b]]) / 20
    a1 = sigmoid(np.dot(inputs, W1) + b1)
    output = sigmoid(np.dot(a1, W2) + b2)
    pred_raw = output[0, 0] * 40
    prediction = round(pred_raw)
    return pred_raw, prediction

def plot_correctness_heatmap(W1, W2, b1, b2):
    correctness = np.zeros((21, 21), dtype=float)
    for i in range(21):
        for j in range(21):
            pred_raw, pred = predict(i, j, W1, W2, b1, b2)
            actual = i + j
            correctness[i, j] = 1.0 if pred == actual else 0.0

    fig, ax = plt.subplots()
    cax = ax.imshow(correctness, origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    fig.colorbar(cax, ax=ax, label="Correct (1) / Incorrect (0)")

    # do this selected_pairs better, also used above in another def
    selected_pairs = st.session_state.training_pairs
    train_x = [pair[1] for pair in selected_pairs]  # j
    train_y = [pair[0] for pair in selected_pairs]  # i
    ax.scatter(train_x, train_y,
               marker='o', s=70, facecolors='none', edgecolors='black',
               label="Training Samples")
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    ax.set_xlabel("Input b")
    ax.set_ylabel("Input a")
    ax.set_title("Heatmap of Correct (1) vs Incorrect (0) Predictions")
    ax.legend(loc="upper right")

    #ax.set_title("Prediction Correctness Heatmap")
    #ax.set_xlabel("Input B")
    #ax.set_ylabel("Input A")
    return fig

def visualize_network_annotated(a, b, W1, W2, b1, b2):
    inputs = np.array([[a, b]]) / 20.0
    z1 = np.dot(inputs, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)
    predicted_sum = output[0, 0] * 40

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_title(f"Annotated Network for Input ({a}, {b}) → Prediction: {round(predicted_sum)}", fontsize=14)

    # --- Layer Coordinates ---
    x_input, x_hidden, x_output = 1, 4, 7
    n_inputs = 2
    n_hidden = len(a1[0])
    y_center = sum(range(n_hidden)) / n_hidden
    y_input = [y_center + 0.5 - i for i in range(n_inputs)]
    y_hidden = list(range(len(a1[0])))
    y_output = [sum(y_hidden) / len(y_hidden)]

    # --- Input Layer ---
    input_vals = [a / 20.0, b / 20.0]
    for i, val in enumerate(input_vals):
        raw_val = [a, b][i]
        ax.scatter(x_input, y_input[i], s=1000, c='skyblue')

        # Main number in the circle: scaled value
        ax.text(x_input, y_input[i], f"{val:.2f}", ha='center', va='center', fontsize=12, weight='bold')

        # Left-side label: Input name, raw value, and scaling info
        ax.text(
            x_input - 0.6, y_input[i],
            f"Input {i+1} = {raw_val}\nscaled = {val:.2f} (÷20)",
            ha='right', va='center', fontsize=ANNOTATION_FONT, family='monospace'
        )

    # --- Hidden Layer ---
    for j, (z_val, a_val) in enumerate(zip(z1[0], a1[0])):
        ax.scatter(x_hidden, y_hidden[j], s=1000, c='lightgreen')
        ax.text(x_hidden, y_hidden[j], f"{a_val:.2f}", ha='center', va='center', fontsize=12)
        ax.text(x_hidden + 0.6, y_hidden[j],
                f"z={z_val:.2f}\na=sig(z)={a_val:.2f}",
                ha='left', va='center', fontsize=ANNOTATION_FONT, family='monospace')

    # --- Output Layer ---
    ax.scatter(x_output, y_output[0], s=1000, c='salmon')
    ax.text(x_output, y_output[0], f"{output[0,0]:.2f}", ha='center', va='center', fontsize=12)
    rounded_pred = round(predicted_sum)
    ax.text(x_output + 0.6, y_output[0],
            f"z={z2[0,0]:.2f}\na=sig(z)={output[0,0]:.2f}\n×40 = {predicted_sum:.2f}\nOutput(rounded) = {rounded_pred}",
            ha='left', va='center', fontsize=ANNOTATION_FONT, family='monospace')

    # --- Connections ---
    for i in range(len(input_vals)):
        for j in range(len(a1[0])):
            ax.plot([x_input, x_hidden], [y_input[i], y_hidden[j]], 'gray', linewidth=0.5)
    for j in range(len(a1[0])):
        ax.plot([x_hidden, x_output], [y_hidden[j], y_output[0]], 'gray', linewidth=0.5)

    # --- Labels ---
    ax.text(x_input, max(y_input)+1, "Inputs", ha='center', fontsize=12)
    ax.text(x_hidden, max(y_hidden)+1, "Hidden Layer", ha='center', fontsize=12)
    ax.text(x_output, max(y_output)+1, "Output", ha='center', fontsize=12)

    plt.tight_layout()
    return fig

########################
# Streamlit UI
st.title("🧠 Neural Network for Learning Addition")

# Initialize session state for training pairs
if "training_pairs" not in st.session_state:
    st.session_state.training_pairs = [(0, 0), (0, 10), (10, 0), (5, 5), (10, 10), (5, 10), (10, 5), (20, 0), (0, 20)]

st.subheader("🧮 Training Data Pairs")

# Display and delete existing pairs
pairs_to_delete = []
for idx, (a, b) in enumerate(st.session_state.training_pairs):
    col1, col2, col3 = st.columns([3, 3, 1])
    col1.write(f"**A:** {a}")
    col2.write(f"**B:** {b}")
    if col3.button("❌", key=f"del_{idx}"):
        pairs_to_delete.append(idx)

# Delete selected pairs
for idx in sorted(pairs_to_delete, reverse=True):
    st.session_state.training_pairs.pop(idx)

# Add new pair
with st.form("add_pair_form", clear_on_submit=True):
    col1, col2, col3 = st.columns([3, 3, 2])
    new_a = col1.number_input("A", min_value=0, max_value=20, key="new_a")
    new_b = col2.number_input("B", min_value=0, max_value=20, key="new_b")
    submitted = col3.form_submit_button("➕ Add")
    if submitted:
        new_pair = (int(new_a), int(new_b))
        if new_pair not in st.session_state.training_pairs:
            st.session_state.training_pairs.append(new_pair)

if st.button("Train Model"):
    X_train, y_train = generate_minimal_training_data()
    W1, W2, b1, b2 = train_model(X_train, y_train)

    # Save model in session state
    st.session_state.model = {
        "W1": W1,
        "W2": W2,
        "b1": b1,
        "b2": b2
    }

    st.success("Model trained successfully!")

# Show heatmap if model is available
if "model" in st.session_state:
    st.subheader("Prediction Heatmap")
    W1 = st.session_state.model["W1"]
    W2 = st.session_state.model["W2"]
    b1 = st.session_state.model["b1"]
    b2 = st.session_state.model["b2"]
    fig = plot_correctness_heatmap(W1, W2, b1, b2)
    st.pyplot(fig)

if "model" in st.session_state:
    st.subheader("🔢 Try Predictions with Sliders")
    a = st.slider("Select input A", 0, 20, 10)
    b = st.slider("Select input B", 0, 20, 5)

    W1 = st.session_state.model["W1"]
    W2 = st.session_state.model["W2"]
    b1 = st.session_state.model["b1"]
    b2 = st.session_state.model["b2"]

    pred_raw, prediction = predict(a, b, W1, W2, b1, b2)
    st.write(f"**Prediction for ({a} + {b}) = {prediction}**")
    st.caption(f"Raw neural net output (scaled): `{pred_raw:.4f}`")

    st.subheader("🧠 Annotated Neural Network Visualization")
    fig = visualize_network_annotated(a, b, W1, W2, b1, b2)
    st.pyplot(fig)