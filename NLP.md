# Tokenizing

Definition: Tokenizing is the process of splitting text into individual units called tokens, which can be words, subwords, or characters.

Example:

Input: "The quick brown fox jumps over the lazy dog."

Tokens: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

# Stemming

Definition: Stemming is the process of reducing words to their base or root form by stripping suffixes and prefixes. It often results in non-standard words.

Example:

Input: ["running", "ran", "runner"]

Stemmed: ["run", "ran", "runner"]

# Lemmatization

Definition: Lemmatization is the process of reducing words to their base or dictionary form (lemma) by considering the context and morphological analysis.

Example:

Input: ["running", "ran", "runner"]

Lemmatized: ["run", "run", "runner"]

# Word Embeddings in NLP

- Word Embeddings are numeric representations of words in a lower-dimensional space, capturing semantic and syntactic information.
Semantic Information
Synonyms: Words like "happy" and "joyful" have similar embeddings due to their similar meanings.
Analogies: "King" is to "queen" as "man" is to "woman" captures semantic relationships.
Syntactic Information
Part of Speech: "Run" (verb) and "jog" (verb) have similar embeddings because they share the same grammatical role.
Word Order: Recognizing that "the cat sits" is correct, whereas "cat the sits" is not.

## Approaches for Text Representation

**One-Hot Encoding**

One-hot encoding is a simple method for representing words in natural language processing (NLP). In this encoding scheme, each word in the vocabulary is represented as a unique vector, where the dimensionality of the vector is equal to the size of the vocabulary. The vector has all elements set to 0, except for the element corresponding to the index of the word in the vocabulary, which is set to 1.


Vocabulary: 

{'mat', 'the', 'bird', 'hat', 'on', 'in', 'cat', 'tree', 'dog'}

Word to Index Mapping: 

{'mat': 0, 'the': 1, 'bird': 2, 'hat': 3, 'on': 4, 'in': 5, 'cat': 6, 'tree': 7, 'dog': 8}

One-Hot Encoded Matrix:

cat: [0, 0, 0, 0, 0, 0, 1, 0, 0]

in: [0, 0, 0, 0, 0, 1, 0, 0, 0]

- One hot size is bigger, so embeddings were invented for dimensionality reduction

One-hot encoding results in high-dimensional vectors, making it computationally expensive and memory-intensive, especially with large vocabularies.

It does not capture semantic relationships between words; each word is treated as an isolated entity without considering its meaning or context.

**Bag of Word (Bow)**

Bag-of-Words (BoW) is a text representation technique representing a document as an unordered set of words and their respective frequencies. It discards the word order and captures the frequency of each word in the document, creating a vector representation.

documents = ["This is the first document.",
              "This document is the second document.",
              "And this is the third one.",
              "Is this the first document?"]

Bag-of-Words Matrix:

[[0 1 1 1 0 0 1 0 1]

 [0 2 0 1 0 1 1 0 1]
 
 [1 0 0 1 1 0 1 1 1]
 
 [0 1 1 1 0 0 1 0 1]]

Vocabulary (Feature Names):

['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']

disadvantages highlight its limitations in capturing certain aspects of language structure and semantics:

BoW ignores the order of words in the document, leading to a loss of sequential information and context making it less effective for tasks where word order is crucial, such as in natural language understanding.
BoW representations are often sparse, with many elements being zero resulting in increased memory requirements and computational inefficiency, especially when dealing with large datasets.

**Term frequency-inverse document frequency (TF-IDF)**

Term Frequency-Inverse Document Frequency, commonly known as TF-IDF, is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents (corpus). It is widely used in natural language processing and information retrieval to evaluate the significance of a term within a specific document in a larger corpus. TF-IDF consists of two components:

Term Frequency (TF): 

Term Frequency measures how often a term (word) appears in a document. It is calculated using the formula:

<img src="https://quicklatex.com/cache3/73/ql_411df6fe05630def9df5f91c0bf9fb73_l3.svg" alt="Formula 1">       

Inverse Document Frequency (IDF): 

Inverse Document Frequency measures the importance of a term across a collection of documents. It is calculated using the formula:

<img src="https://quicklatex.com/cache3/41/ql_4877a2e3d394875c97463a5d34c67d41_l3.svg" alt="Formula 2">

The TF-IDF score for a term t in a document d is then given by multiplying the TF and IDF values:

<img src="https://quicklatex.com/cache3/dd/ql_37a35895ccca113a353a775a79ddd8dd_l3.svg" alt="Formula 3">

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
]

Document 1:

dog: 0.3404110310756642

lazy: 0.3404110310756642

over: 0.3404110310756642

jumps: 0.3404110310756642

fox: 0.3404110310756642

brown: 0.3404110310756642

quick: 0.3404110310756642

the: 0.43455990318254417

Document 2:

step: 0.3535533905932738

single: 0.3535533905932738

with: 0.3535533905932738

begins: 0.3535533905932738

miles: 0.3535533905932738

thousand: 0.3535533905932738

of: 0.3535533905932738

journey: 0.3535533905932738

TF-IDF is a widely used technique in information retrieval and text mining, but its limitations should be considered, especially when dealing with tasks that require a deeper understanding of language semantics. For example:

TF-IDF treats words as independent entities and doesn’t consider semantic relationships between them. This limitation hinders its ability to capture contextual information and word meanings.

Sensitivity to Document Length: Longer documents tend to have higher overall term frequencies, potentially biasing TF-IDF towards longer documents.


## Neural Approach

**Word2Vec**

The underlying idea is that words with similar meanings should have similar vector representations. In Word2Vec every word is assigned a vector. We start with either a random vector or one-hot vector.

There are two neural embedding methods for Word2Vec, Continuous Bag of Words (CBOW) and Skip-gram.

**1.Continuous Bag of Words(CBOW)**

Continuous Bag of Words (CBOW) is a type of neural network architecture used in the Word2Vec model. The primary objective of CBOW is to predict a target word based on its context, which consists of the surrounding words in a given window. Given a sequence of words in a context window, the model is trained to predict the target word at the center of the window.

**2.Skip-Gram**

The Skip-Gram model learns distributed representations of words in a continuous vector space. The main objective of Skip-Gram is to predict context words (words surrounding a target word) given a target word. This is the opposite of the Continuous Bag of Words (CBOW) model, where the objective is to predict the target word based on its context. It is shown that this method produces more meaningful embeddings.

- The similarity between two vectors in an inner product space is measured by cosine similarity.

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

# Transformers


- The Transformer is a neural network architecture 
- The key innovation of the Transformer is the use of self-attention mechanisms to process sequences of data, which allows for more parallelization compared to traditional RNNs and LSTMs.

## Encoders-Decoders

There are three main blocks in the encoder-decoder model,

Encoder
Hidden Vector
Decoder

The Encoder will convert the input sequence into a single-dimensional vector (hidden vector). 

The decoder will convert the hidden vector into the output sequence.

## Encoder Vector
This is the final hidden state produced from the encoder part of the model.
This vector aims to encapsulate the information for all input elements in order to help the decoder make accurate predictions.
It acts as the initial hidden state of the decoder part of the model.

##Decoder
The Decoder generates the output sequence by predicting the next output Yt given the hidden state ht.
