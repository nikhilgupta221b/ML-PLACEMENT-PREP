# RNN vs LSTM vs GRU

## Recurrent Neural Networks (RNN)
Recurrent Neural Network(RNN) is a type of Neural Network where the output from the previous step is fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other. Still, in cases when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. Thus RNN came into existence, which solved this issue with the help of a Hidden Layer. The main and most important feature of RNN is its Hidden state, which remembers some information about a sequence. The state is also referred to as Memory State since it remembers the previous input to the network. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output. This reduces the complexity of parameters, unlike other neural networks.
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
The LSTM architectures involves the memory cell which is controlled by three gates: the input gate, the forget gate, and the output gate. These gates decide what information to add to, remove from, and output from the memory cell.

The input gate controls what information is added to the memory cell.
The forget gate controls what information is removed from the memory cell.
The output gate controls what information is output from the memory cell

This allows LSTM networks to selectively retain or discard information as it flows through the network, which allows them to learn long-term dependencies.

The LSTM maintains a hidden state, which acts as the short-term memory of the network. The hidden state is updated based on the input, the previous hidden state, and the memory cell’s current state.

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
The basic idea behind GRU is to use gating mechanisms to selectively update the hidden state of the network at each time step. The gating mechanisms are used to control the flow of information in and out of the network. The GRU has two gating mechanisms, called the reset gate and the update gate.

The reset gate determines how much of the previous hidden state should be forgotten, while the update gate determines how much of the new input should be used to update the hidden state. The output of the GRU is calculated based on the updated hidden state.
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
