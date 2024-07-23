# RNN vs LSTM vs GRU

## Recurrent Neural Networks (RNN)
- **Architecture**:
  - Simple, single-layer structure where each unit has:
    - Input from the current time step.
    - Recurrent connection from the previous time step’s hidden state.
  - Hidden state updated as:
    <pre><code>h_t = tanh(W_h * x_t + U_h * h_{t-1} + b_h)</code></pre>
- **Advantages**:
  - Easy to implement.
  - Suitable for short sequences and tasks where dependencies are short-term.
- **Disadvantages**:
  - Suffers from vanishing and exploding gradient problems.
  - Struggles with long-term dependencies.
- **Use Cases**:
  - Simple sequence prediction.
  - Basic time series analysis.

## Long Short-Term Memory (LSTM)
- **Architecture**:
  - Complex structure with memory cells and three gates: 
    - **Forget Gate**:
      <pre><code>f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)</code></pre>
    - **Input Gate**:
      <pre><code>i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)</code></pre>
    - **Output Gate**:
      <pre><code>o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)</code></pre>
    - **Candidate Memory Cell**:
      <pre><code>~C_t = tanh(W_C * [h_{t-1}, x_t] + b_C)</code></pre>
    - **Cell State Update**:
      <pre><code>C_t = f_t * C_{t-1} + i_t * ~C_t</code></pre>
    - **Hidden State**:
      <pre><code>h_t = o_t * tanh(C_t)</code></pre>
- **Advantages**:
  - Effectively captures long-term dependencies.
  - Mitigates vanishing and exploding gradient problems.
- **Disadvantages**:
  - Computationally expensive and slower to train.
  - More complex implementation.
- **Use Cases**:
  - Language modeling and machine translation.
  - Speech recognition.
  - Time series forecasting.

## Gated Recurrent Unit (GRU)
- **Architecture**:
  - Simplified structure with two gates: 
    - **Reset Gate**:
      <pre><code>r_t = sigmoid(W_r * [h_{t-1}, x_t] + b_r)</code></pre>
    - **Update Gate**:
      <pre><code>z_t = sigmoid(W_z * [h_{t-1}, x_t] + b_z)</code></pre>
    - **Candidate Hidden State**:
      <pre><code>~h_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)</code></pre>
    - **Final Hidden State**:
      <pre><code>h_t = z_t * h_{t-1} + (1 - z_t) * ~h_t</code></pre>
- **Advantages**:
  - Faster and computationally more efficient than LSTM.
  - Also captures long-term dependencies effectively.
- **Disadvantages**:
  - Slightly less expressive than LSTMs for some tasks.
  - Fewer hyperparameters to tune.
- **Use Cases**:
  - Similar to LSTM, including language modeling, speech recognition, and time series forecasting.
  - Preferable when computational efficiency is important.

## Summary
- **RNN**:
  - Simple and effective for short sequences.
  - Struggles with long-term dependencies.

- **LSTM**:
  - Powerful for capturing long-term dependencies.
  - More complex and slower to train.

- **GRU**:
  - Simplified and faster alternative to LSTM.
  - Efficient for both short and long sequences.
