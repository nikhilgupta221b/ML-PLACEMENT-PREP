# Transformer Architectures: Encoder and Decoder

## Encoder

- **Input Processing**: The encoder takes in the raw input data (e.g., a sequence of words in a sentence for NLP tasks or image patches for vision tasks) and processes it to create embeddings.
- **Embedding Creation**: The input is transformed into a fixed-dimensional vector, often using embeddings that convert discrete tokens into continuous vectors.
- **Positional Encoding**: Since transformers lack an inherent notion of order, positional encodings are added to the input embeddings to provide information about the position of each token in the sequence.
- **Self-Attention and Feed-Forward Networks**: Through multiple layers of self-attention and feed-forward networks, the encoder processes the input embeddings to capture dependencies and relationships between the tokens, resulting in contextually rich representations (embeddings).

## Decoder

- **Input Preparation**: The decoder takes the output embeddings from the encoder and the target sequence (shifted right for training purposes in NLP tasks) as inputs.
- **Masked Self-Attention**: The decoder uses masked self-attention to process the target sequence. Masking ensures that the prediction for a particular position only depends on known outputs and not future positions.
- **Cross-Attention**: The decoder employs cross-attention to attend to the encoder’s output embeddings. This step allows the decoder to incorporate information from the entire input sequence when making predictions.
- **Feed-Forward Networks**: Similar to the encoder, the decoder has feed-forward networks that further process the attended embeddings to generate output predictions.
- **Output Generation**: The final layer typically maps the processed embeddings to the desired output format, such as probability distributions over a vocabulary for NLP tasks or bounding boxes and class labels for object detection tasks.

## Example in NLP (e.g., Translation)

### Encoder

- **Input Sentence**: "The cat sits on the mat."
- **Embeddings + Positional Encodings**: Vector representations of each word plus positional information.
- **Self-Attention Layers**: Capture dependencies like "cat" and "sits" being closely related.
- **Encoder Output**: Contextual embeddings for the entire sentence.

### Decoder

- **Target Sentence (Shifted Right)**: "<start> Le chat est assis sur le tapis."
- **Masked Self-Attention**: Process target sequence to predict next word.
- **Cross-Attention**: Attend to encoder’s output embeddings to inform prediction.
- **Output Generation**: Predict the next word in the target language.

## Example in Vision (e.g., DETR)

### Encoder

- **Input Image**: An image split into patches.
- **Embeddings + Positional Encodings**: Vector representations of image patches plus positional information.
- **Self-Attention Layers**: Capture dependencies between different parts of the image.
- **Encoder Output**: Contextual embeddings for the image patches.

### Decoder

- **Object Queries**: Learnable positional encodings representing potential objects.
- **Masked Self-Attention**: Process object queries.
- **Cross-Attention**: Attend to encoder’s output embeddings to inform detection.
- **Output Generation**: Predict bounding boxes and class labels for objects in the image.

## Summary

- **Encoder**: Converts input data into rich, contextually informed embeddings using self-attention and feed-forward layers.
- **Decoder**: Uses these embeddings, along with target sequence information, to generate predictions, employing masked self-attention, cross-attention, and feed-forward layers.
