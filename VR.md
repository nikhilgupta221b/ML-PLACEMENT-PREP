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

### Fixed Masking (BERT)
- **Masking Pattern**: Fixed positions for masked words.
- **Consistency**: Same words are masked at the same positions in each epoch.
- **Training Focus**: Learns to predict masked words based on fixed context.

### Dynamic Masking (RoBERTa)
- **Masking Pattern**: Variable positions for masked words.
- **Variability**: Different words are masked each time a sequence is fed through the model.
- **Training Focus**: Provides diverse contexts for learning, improving generalization.

### Benefits of Dynamic Masking
- **Increased Variety**: Exposes the model to a broader range of masked positions.
- **Improved Learning**: Helps the model to generalize better and avoid overfitting.
- **Better Representation**: Leads to more nuanced and robust contextual understanding.


# Definitions and Working of Various Vision and Transformer Models

## DeiT (Data-efficient Image Transformer)

**Definition:**
DeiT stands for Data-efficient Image Transformer. It is a vision transformer model designed to be more efficient with data and computational resources.

**Working:**
- **Architecture:** DeiT uses the Transformer architecture, originally designed for NLP tasks, for image classification.
- **Efficiency:** It introduces a training strategy that uses data augmentation and knowledge distillation to achieve high performance with limited data and computational resources.
- **Key Feature:** Employs a teacher-student framework where a large teacher model guides a smaller student model, enabling effective learning with fewer data.

## DINO (Self-Distillation with No Labels)

**Definition:**
DINO (Self-Distillation with No Labels) is a self-supervised learning method for visual representations.

**Working:**
- **Architecture:** DINO uses a vision transformer model and operates in a self-supervised manner without relying on labeled data.
- **Self-Distillation:** It applies self-distillation where the model learns representations by comparing different augmented views of the same image, encouraging consistency in the learned features.
- **Key Feature:** Leverages data augmentation and a momentum encoder to learn robust features without requiring labeled data.

## Shifted Window Vision Transformer

**Definition:**
Shifted Window Vision Transformer is a variant of the Vision Transformer that addresses the limitations of the standard window-based attention mechanism.

**Working:**
- **Architecture:** Utilizes a window-based self-attention mechanism, where attention is computed within local windows.
- **Shifted Windows:** Implements a shifted window approach where windows are shifted between layers to capture cross-window interactions and global context.
- **Key Feature:** Enhances the ability to model global dependencies and improve performance on vision tasks.

## Non-Maximal Suppression (NMS)

**Definition:**
Non-Maximal Suppression (NMS) is a post-processing technique used in object detection to filter overlapping bounding boxes.

**Working:**
- **Procedure:** Given a set of bounding boxes with associated confidence scores, NMS removes redundant boxes by keeping only the one with the highest score while suppressing others that overlap significantly.
- **Key Feature:** Helps to refine detection results by eliminating duplicate detections and retaining the most accurate bounding box for each object.

A low NMS threshold means that NMS will be stricter about the amount of overlap allowed between bounding boxes before suppressing one of them

A high NMS threshold means that NMS will tolerate more overlap between bounding boxes before considering one of them redundant.

## DETR (DEtection TRansformer)

**Definition:**
DETR (Detection Transformer) is a Transformer-based model designed for object detection tasks.

**Working:**
- **Architecture:** Combines a vision backbone with a Transformer architecture to directly predict bounding boxes and class labels.
- **End-to-End Learning:** DETR performs object detection in an end-to-end manner, eliminating the need for region proposal networks (RPNs) and anchor boxes.
- **Key Feature:** Utilizes a set-based global loss to ensure a one-to-one mapping between predicted and ground-truth objects, improving detection accuracy.

## ViLBERT (Vision-and-Language BERT)

**Definition:**
ViLBERT (Vision-and-Language BERT) is a model that extends BERT (Bidirectional Encoder Representations from Transformers) to handle both visual and textual information.


**Working:**
- **Architecture:** Uses separate streams for processing visual and textual inputs, which are later fused using cross-attention mechanisms.
- **Multimodal Learning:** Trains on tasks that require understanding both visual and textual information, such as visual question answering.
- **Key Feature:** Allows joint representation learning from images and text, enabling the model to understand and generate responses based on multimodal inputs.


### Wu-Palmer Similarity

- **Concept**: Measures semantic similarity based on shared hierarchical relationships in a lexical taxonomy (e.g., WordNet).
- **Taxonomy**: Uses lexical databases where words are organized into synsets with hypernym (general term) and hyponym (specific term) relationships.
- **Process**:
  1. **Find Common Ancestor**: Identify the most specific common ancestor (least common subsumer) of the two words.
  2. **Compute Similarity**:
     \[
     \text{Similarity}(w_1, w_2) = \frac{2 \cdot \text{Depth}_{LCS}}{\text{Depth}_{w_1} + \text{Depth}_{w_2} + 2 \cdot \text{Depth}_{LCS}}
     \]
     - \(\text{Depth}_{LCS}\): Depth of the least common subsumer.
     - \(\text{Depth}_{w_1}\): Depth of word \(w_1\).
     - \(\text{Depth}_{w_2}\): Depth of word \(w_2\).

- **Example**: For "dog" and "cat":
  - Common ancestor: "mammal."
  - Assume depths: "dog" = 5, "cat" = 5, "mammal" = 3.
  - Similarity:
    \[
    \text{Similarity}(\text{"dog"}, \text{"cat"}) = \frac{2 \cdot 3}{5 + 5 + 2 \cdot 3} = 0.375
    \]

- **Benefits**:
  - **Semantics-Based**: Reflects hierarchical semantic relationships.
  - **Intuitive**: Captures similarity based on shared ancestors.

- **Limitations**:
  - **Taxonomy Dependency**: Relies on the structure of the lexical database.
  - **Static**: May not adapt to evolving language and meanings.


## How ViT and BERT were Combined for Visual QnA

### 1. Image Encoding using ViT:
- **ViT** takes the input image, splits it into patches, and treats each patch as a sequence element (similar to how words are processed in text).
- These patches are fed through multiple transformer layers, which capture relationships between different parts of the image.
- The output of ViT is a sequence of **image embeddings** that represent the visual information of the image in a format understandable by a transformer model.

### 2. Text Encoding using BERT:
- **BERT** is used to process the question. It tokenizes the input question and passes it through transformer layers to generate a sequence of embeddings for the words in the question.
- BERT’s output is a **contextualized representation** of the question, where each token’s embedding captures its meaning in the context of the entire question.

### 3. Fusion of Vision and Textual Features:
- The outputs from ViT (**image embeddings**) and BERT (**question embeddings**) are combined in a way that allows the model to correlate information between the image and the question.
- Common methods for combining the two modalities include:
  - **Concatenation**: Image embeddings and question embeddings are concatenated and then passed through additional transformer layers to allow the model to learn how to relate the two.
  - **Cross-attention**: The model uses an attention mechanism where the image features attend to the question features, and vice versa, allowing the model to learn which parts of the image are relevant to the question.

### 4. Answer Prediction:
- After combining the image and text embeddings, the **fused representation** is passed through a series of layers (such as dense layers) that ultimately generate a prediction.
- The output is typically a **classification layer** where the model predicts the most likely answer to the question based on the image.


## LoRA (Low-Rank Adaptation) Explanation

### Overview:
LoRA (Low-Rank Adaptation) is a fine-tuning technique designed to reduce the **computational cost** and **memory requirements** of fine-tuning large models like transformers (e.g., BERT, ViT). It achieves this by introducing **low-rank matrices** that are optimized during training while **freezing** most of the original model's parameters.

---

### How LoRA Works:
1. **Freezing Model Weights**:
   - In LoRA, the **main model weights** are frozen, meaning they are not updated during fine-tuning. These frozen weights are the pre-trained parameters from a model like BERT or ViT.
   - Freezing helps preserve the original knowledge of the model while reducing the number of parameters that need to be optimized.

2. **Introducing Low-Rank Matrices**:
   - Instead of updating the full parameter set, LoRA introduces **low-rank matrices** to learn the specific task features. These low-rank matrices are much smaller in size compared to the full model, leading to **fewer parameters** to update.
   - The low-rank matrices are added to the original model in a way that allows them to capture important task-specific information without the need for full re-training of the large model.

### Significance of Rank:
- The **rank** in LoRA refers to the dimensionality of the low-rank matrices introduced during fine-tuning.
- **Higher rank** means the low-rank matrix can capture more complex relationships, but it also increases the computational cost.
- **Lower rank** reduces the computational cost but may limit the capacity of the model to learn fine-grained details.

In your project, using a **LoRA rank of 32** means that the low-rank matrices had a dimension of 32, which is typically a good trade-off between **performance** (ability to learn) and **efficiency** (low computational cost).

### Advantages of LoRA:
1. **Parameter Efficiency**:
   - Only a small number of additional parameters are learned (the low-rank matrices), leading to significant memory savings.

2. **Scalability**:
   - LoRA allows fine-tuning of very large models (like BERT or ViT) without needing a large amount of computational resources, making it suitable for systems with limited capacity.

3. **Fast Fine-Tuning**:
   - Because only the low-rank matrices are being optimized, the training process is **faster** compared to fine-tuning the full set of parameters in the original model.

### Frozen Model Weights:
- When the model's weights are **frozen**, they are **not updated** during the training process. This helps maintain the pre-trained knowledge that the model has learned during its large-scale training on diverse data.
- LoRA's low-rank matrices work as **additional parameters** that get updated, allowing the model to adapt to the specific task without needing to modify the original weights.

By freezing most of the weights and only updating a small number of parameters (through low-rank matrices), LoRA strikes an **optimal balance** between **adaptability** and **efficiency**.
