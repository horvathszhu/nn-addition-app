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

def visualize_network_annotated(a, b, W1, W2, b1, b2, show_details=True, skeleton_only=False):
    inputs = np.array([[a, b]]) / 20.0
    z1 = np.dot(inputs, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)
    predicted_sum = output[0, 0] * 40

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    if not skeleton_only:
        ax.set_title(f"Annotated Network for Input ({a}, {b}) → Prediction: {round(predicted_sum)}", fontsize=14, pad=60)

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
        ax.scatter(x_input, y_input[i], s=1000, c='skyblue')
        if not skeleton_only:
            raw_val = [a, b][i]
            ax.text(x_input, y_input[i], f"{val:.2f}", ha='center', va='center', fontsize=12, weight='bold')
            ax.text(
                x_input - 0.3, y_input[i],
                f"Input {i+1} = {raw_val}\nscaled = {val:.2f} (÷20)",
                ha='right', va='center', fontsize=ANNOTATION_FONT, family='monospace'
            )

    # --- Hidden Layer ---
    topmost_hidden_index = np.argmax(y_hidden)
    for j, (z_val, a_val) in enumerate(zip(z1[0], a1[0])):
        ax.scatter(x_hidden, y_hidden[j], s=1000, c='lightgreen')
        if not skeleton_only:
            ax.text(x_hidden, y_hidden[j], f"{a_val:.2f}", ha='center', va='center', fontsize=12)
            z_parts = [f"{W1[i, j]:.2f}×{input_vals[i]:.2f}" for i in range(len(input_vals))]
            z_eq = " + ".join(z_parts) + f" + {b1[0, j]:.2f}"
            z_text = f"z = {z_eq} = {z1[0, j]:.2f}\na = sig(z) = {a1[0, j]:.2f}" if (show_details and j == topmost_hidden_index) else f"z = {z1[0, j]:.2f}\na = {a1[0, j]:.2f}"
            ax.text(
                x_hidden + 0.3, y_hidden[j],
                z_text,
                ha='left', va='center',
                fontsize=ANNOTATION_FONT,
                family='monospace'
            )

    # --- Output Layer ---
    ax.scatter(x_output, y_output[0], s=1000, c='salmon')
    if not skeleton_only:
        ax.text(x_output, y_output[0], f"{output[0,0]:.2f}", ha='center', va='center', fontsize=12)
        rounded_pred = round(predicted_sum)
        ax.text(x_output + 0.3, y_output[0],
                f"z={z2[0,0]:.2f}\na=sig(z)={output[0,0]:.2f}\n×40 = {predicted_sum:.2f}\nOutput(rounded) = {rounded_pred}",
                ha='left', va='center', fontsize=ANNOTATION_FONT, family='monospace')

    # --- Connections ---
    for i in range(len(input_vals)):
        for j in range(len(a1[0])):
            ax.plot([x_input, x_hidden], [y_input[i], y_hidden[j]], 'gray', linewidth=0.5)
            if not skeleton_only:
                weight_val = W1[i, j]
                mid_x = x_input + (x_hidden - x_input) * (2/3)
                mid_y = y_input[i] + (y_hidden[j] - y_input[i]) * (2/3)
                ax.text(mid_x, mid_y, f"{weight_val:.2f}", fontsize=ANNOTATION_FONT, ha='center', va='center', color='darkblue')
    for j in range(len(a1[0])):
        ax.plot([x_hidden, x_output], [y_hidden[j], y_output[0]], 'gray', linewidth=0.5)
        if not skeleton_only:
            weight_val = W2[j, 0]
            mid_x = x_hidden + (x_output - x_hidden) * (2/3)
            mid_y = y_hidden[j] + (y_output[0] - y_hidden[j]) * (2/3)
            ax.text(mid_x, mid_y, f"{weight_val:.2f}", fontsize=ANNOTATION_FONT, ha='center', va='center', color='darkred')

    # --- Labels ---
    if not skeleton_only:
        ax.text(x_input, max(y_hidden)+0.5, "Inputs", ha='center', fontsize=ANNOTATION_FONT)
        ax.text(x_hidden, max(y_hidden)+0.5, "Hidden Layer", ha='center', fontsize=ANNOTATION_FONT)
        ax.text(x_output, max(y_hidden)+0.5, "Output", ha='center', fontsize=ANNOTATION_FONT)

    plt.tight_layout()
    return fig

def styled_caption(text, font_size=18):
    st.markdown(
        f"<div style='color: black; font-size: {font_size}px; line-height: 1.6; margin: 0;'>{text}</div>",
        unsafe_allow_html=True
    )


########################
# Streamlit UI
st.title("🧠 Neural Network for Learning Addition")

styled_caption(
    """

This simple neural net learns to add two integers between 0 and 20, 
based on just a few training examples.

The network takes the input pairs `(a, b)`, feeds them through its internal calculations, 
and compares the output with the actual sum `a + b`.

If there's an error (which is usually the case initially), the model adjusts its internal 
weights to reduce that error. This adjustment is done repeatedly for each training pair.

The network has a simple 3-layer structure:
- <span style='color:skyblue'><b>Input layer</b></span>: 2 nodes (for `a` and `b`)
- <span style='color:lightgreen'><b>Hidden layer</b></span>: 5 nodes (a small but sufficient number for learning this task)
- <span style='color:salmon'><b>Output layer</b></span>: 1 node (predicts the sum)
""")

W1_dummy, W2_dummy, b1_dummy, b2_dummy = initialize_weights(2, 5, 1)
fig = visualize_network_annotated(10, 10, W1_dummy, W2_dummy, b1_dummy, b2_dummy, skeleton_only=True)
st.pyplot(fig)

styled_caption(
    """

**Other parameters:**
- `learning_rate = 0.1`: controls how much the weights are adjusted during training
- `n_epochs = 10000`: each training pair is shown to the network 10,000 times
""")

# Initialize session state for training pairs
if "training_pairs" not in st.session_state:
    st.session_state.training_pairs = [(0, 0), (0, 10), (10, 0), (5, 5), (10, 10), (5, 10), (10, 5), (20, 0), (0, 20)]

# Keep space from previous section
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("🧮 Training Data Pairs")
styled_caption("""

Here are the number pairs the model will learn from. Add new examples or remove existing ones.
""")

# Display and delete existing pairs
pairs_to_delete = []
for idx, (a, b) in enumerate(st.session_state.training_pairs):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"**(a={a}, b={b})**")
    with col2:
        if st.button("❌", key=f"del_{idx}"):
            pairs_to_delete.append(idx)

# Delete selected pairs and rerun immediately to update UI
if pairs_to_delete:
    for idx in sorted(pairs_to_delete, reverse=True):
        st.session_state.training_pairs.pop(idx)
    st.rerun()

# Add new pair
with st.form("add_pair_form", clear_on_submit=True):
    col1, col2, col3 = st.columns([3, 3, 2])
    new_a = col1.number_input("a", min_value=0, max_value=20, key="new_a")
    new_b = col2.number_input("b", min_value=0, max_value=20, key="new_b")
    submitted = col3.form_submit_button("➕ Add")
    if submitted:
        new_pair = (int(new_a), int(new_b))
        if new_pair not in st.session_state.training_pairs:
            st.session_state.training_pairs.append(new_pair)
            st.rerun()

# Keep space from previous section
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("🏋️ Train the Model")
styled_caption("""&nbsp;

Click the button below to train the neural network using your selected data pairs. To try out other numbers, go back to the pairs, change them and push again to retrain.&nbsp;<br><br>
""")

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

    # Keep space from previous section
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Prediction Heatmap")
    styled_caption("""&nbsp;

See how well the model performs across all possible number pairs. Green means correct, red means incorrect.
""")
    W1 = st.session_state.model["W1"]
    W2 = st.session_state.model["W2"]
    b1 = st.session_state.model["b1"]
    b2 = st.session_state.model["b2"]
    fig = plot_correctness_heatmap(W1, W2, b1, b2)
    st.pyplot(fig)

if "model" in st.session_state:
    st.subheader("🔢 Try Predictions with Sliders")
    styled_caption("""&nbsp;

Test the trained model by selecting values for A and B. The model will predict their sum.
""")
    a = st.slider("Select input a", 0, 20, 10)
    b = st.slider("Select input b", 0, 20, 5)

    W1 = st.session_state.model["W1"]
    W2 = st.session_state.model["W2"]
    b1 = st.session_state.model["b1"]
    b2 = st.session_state.model["b2"]

    pred_raw, prediction = predict(a, b, W1, W2, b1, b2)
    st.write(f"**Prediction for ({a} + {b}) = {prediction}** (Raw neural net output (scaled): `{pred_raw:.4f}`)")
#    styled_caption(f"""&nbsp;
#
#Raw neural net output (scaled): `{pred_raw:.4f}`
#""")

    # Keep space from previous section
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🧠 Annotated Neural Network Visualization")

    # ---------- Picking calculated actual z equation values for the caption text ---------- 
    # Pick the topmost hidden neuron based on layout
    n_hidden = W1.shape[1]
    y_hidden = list(range(n_hidden))
    top_idx = np.argmax(y_hidden)  # visually topmost neuron

    inputs_scaled = np.array([a, b]) / 20.0
    weights = W1[:, top_idx]
    bias = b1[0, top_idx]
    z_val = np.dot(inputs_scaled, weights) + bias

    z_terms = [f"{weights[i]:.2f}×{inputs_scaled[i]:.2f}" for i in range(len(inputs_scaled))]
    z_eq_str = " + ".join(z_terms) + f" + {bias:.2f} = {z_val:.2f}"
    a_val = sigmoid(z_val)

    styled_caption(f"""&nbsp;

This diagram illustrates how the neural network processes your selected inputs (a and b) to generate a prediction.


Each circle represents a neuron in the input, hidden, or output layer. The values inside the circles are the neuron activations: the neutron "fires" with about that strength. Input neurons show scaled `a` and `b` (numbers divided by 20). Scaling keeps them between 0 and 1 helping the network to learn better. Hidden and output neurons show the result of the sigmoid function applied to the weighted sum `z`.

The sigmoid function squashes values into a range between 0 and 1. It’s defined as `sig(z) = 1 / (1 + exp(-z))`. 
This introduces non-linearity making the network learn interesting and complex patterns. There are other functions neural networks use. 


Lines connecting neurons show how information flows through the network. Numbers on these lines represent the actual learned weights (w) — they determine how strongly each input influences the next layer. Blue weights go from input to hidden, red weights go from hidden to output.

This annotated view helps you trace the full computation path from raw inputs to final prediction.
For each neuron, a weighted sum `z` is computed as: `z = w₁·x₁ + w₂·x₂ + ... + b`, followed by the sigmoid activation: `a = sig(z)`.

Below is the actual equation for the top hidden neuron for your current input (To reduce clutter, only this top neuron shows the full equation. Others show just the final `z` and `a` values.):
    """)

    st.code(f"z = {z_eq_str}\nsig(z) = {a_val:.2f}", language="python")

    fig = visualize_network_annotated(a, b, W1, W2, b1, b2)
    st.pyplot(fig)