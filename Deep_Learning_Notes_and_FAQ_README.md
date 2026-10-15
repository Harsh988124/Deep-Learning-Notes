# Deep Learning — Detailed Notes & Interview FAQ

A beginner-to-interview reference for learning Deep Learning, revising core concepts, and preparing for technical interviews.

## Table of Contents

1. [Learning Objectives](#learning-objectives)
2. [Prerequisites](#prerequisites)
3. [What Is Deep Learning?](#what-is-deep-learning)
4. [AI vs. ML vs. Deep Learning](#ai-vs-ml-vs-deep-learning)
5. [How a Neural Network Works](#how-a-neural-network-works)
6. [Neurons, Layers, Weights, and Biases](#neurons-layers-weights-and-biases)
7. [Activation Functions](#activation-functions)
8. [Forward Propagation](#forward-propagation)
9. [Loss Functions](#loss-functions)
10. [Backpropagation and Gradients](#backpropagation-and-gradients)
11. [Optimization Algorithms](#optimization-algorithms)
12. [Training, Validation, and Test Sets](#training-validation-and-test-sets)
13. [Overfitting and Regularization](#overfitting-and-regularization)
14. [Important Training Concepts](#important-training-concepts)
15. [Common Neural Network Architectures](#common-neural-network-architectures)
16. [CNNs](#convolutional-neural-networks-cnns)
17. [RNNs, LSTMs, and GRUs](#rnns-lstms-and-grus)
18. [Transformers and Attention](#transformers-and-attention)
19. [Autoencoders and GANs](#autoencoders-and-gans)
20. [Transfer Learning and Fine-Tuning](#transfer-learning-and-fine-tuning)
21. [Evaluation Metrics](#evaluation-metrics)
22. [Practical Workflow](#practical-workflow)
23. [Python Example: A Tiny Neural Network](#python-example-a-tiny-neural-network)
24. [Deep Learning Interview FAQ](#deep-learning-interview-faq)
25. [Practice Questions](#practice-questions)
26. [Glossary](#glossary)
27. [Learning Resources](#learning-resources)

---

## Learning Objectives

After studying these notes, you should be able to:

- Explain neural networks and the main deep learning training loop.
- Choose common activation functions, losses, optimizers, and evaluation metrics.
- Explain backpropagation, gradient descent, overfitting, and regularization.
- Distinguish CNNs, RNNs, Transformers, autoencoders, and GANs.
- Build a baseline model, evaluate it correctly, and diagnose common failures.
- Answer common entry-level deep learning interview questions.

## Prerequisites

Recommended foundations:

- **Python:** functions, classes, loops, list/dictionary operations, virtual environments.
- **NumPy:** arrays, broadcasting, matrix multiplication, vectorization.
- **Math:** functions, derivatives, chain rule, vectors, matrices, probability, mean and variance.
- **Machine learning:** train/test split, classification, regression, overfitting, metrics.

You do not need to master every mathematical proof before building your first model. Learn the intuition, implement a small example, then revisit the math.

## What Is Deep Learning?

Deep learning is a branch of machine learning that uses neural networks with multiple processing layers to learn representations from data. Instead of manually designing every feature, a sufficiently trained network can learn useful intermediate representations from examples.

Examples:
- Image classification and object detection
- Speech recognition and audio processing
- Language translation, summarization, and chat assistants
- Recommendation and anomaly detection
- Medical-image analysis and time-series forecasting

**Strengths:** learns complex patterns, can work well with large unstructured datasets, and can learn features automatically.

**Limitations:** may need large datasets and substantial compute, can be difficult to interpret, may inherit data bias, and can fail when inputs differ from training data. Deep learning is not automatically better than simpler models on small, structured datasets.

## AI vs. ML vs. Deep Learning

- **Artificial Intelligence (AI):** the broad field of systems performing tasks associated with intelligence.
- **Machine Learning (ML):** systems learn patterns from data rather than relying only on explicitly written rules.
- **Deep Learning (DL):** ML based on neural networks with multiple layers.

A useful hierarchy is: Deep Learning is a subset of Machine Learning, which is a subset of AI.

## How a Neural Network Works

A neural network maps inputs to outputs through connected layers. During training it:
1. Receives a batch of input examples.
2. Performs a forward pass to produce predictions.
3. Computes a loss that measures prediction error.
4. Calculates gradients using backpropagation.
5. Updates trainable parameters with an optimizer.
6. Repeats for many batches and epochs.
7. Evaluates performance on held-out data.

The goal is to learn parameters that generalize to unseen examples, not merely memorize the training set.

## Neurons, Layers, Weights, and Biases

A basic neuron computes a weighted sum and then applies an activation:

`z = w1*x1 + w2*x2 + ... + wn*xn + b`

`a = activation(z)`

- **Input layer:** receives the features.
- **Hidden layers:** transform inputs into intermediate representations.
- **Output layer:** produces the prediction.
- **Weights:** control the influence of input signals.
- **Bias:** lets a neuron shift its activation independently of the weighted input.
- **Parameters:** values learned during training, typically weights and biases.
- **Hyperparameters:** settings selected by the practitioner, such as learning rate, batch size, number of layers, and hidden units.

For a dense layer with `n_in` inputs and `n_out` outputs, the number of parameters is `n_in * n_out + n_out` (weights plus one bias per output unit).

## Activation Functions

Activation functions introduce non-linearity. Without non-linear activations, stacking ordinary linear layers still produces a linear transformation and cannot represent many complex relationships.

| Function | Intuition / formula | Typical use | Caution |
|---|---|---|---|
| Sigmoid | `1 / (1 + exp(-x))`; output in (0, 1) | Binary probability output | Saturation and vanishing gradients |
| Tanh | Output in (-1, 1) | Some recurrent-network components | Can saturate |
| ReLU | `max(0, x)` | Common hidden-layer default | Units can become inactive |
| Leaky ReLU | Small negative slope for x < 0 | Alternative to ReLU | Extra slope choice |
| Softmax | Converts logits to a distribution summing to 1 | Mutually exclusive multiclass output | Usually paired with cross-entropy |

**Practical rule:** ReLU or a related function is common in hidden layers; sigmoid is common for binary classification outputs; softmax is common for single-label multiclass classification. Frameworks often combine output logits and loss in a numerically stable operation, so follow the framework's expected input format.

## Forward Propagation

Forward propagation computes a prediction from input to output.

For one dense layer:
- `z = W x + b`
- `a = f(z)`

For a network, the output of one layer becomes the input to the next. During inference, a trained model performs the forward pass without updating its parameters.

## Loss Functions

A loss function quantifies how far predictions are from targets. Training attempts to minimize the chosen loss.

### Mean Squared Error (MSE)

`MSE = (1/N) * sum((y_true - y_pred)^2)`

Common for regression. Large errors receive a stronger penalty because the difference is squared.

### Binary Cross-Entropy

For binary targets `y` and predicted probability `p`:

`L = -[y log(p) + (1-y) log(1-p)]`

Common for binary classification. Use the numerically stable logits-based version where available.

### Categorical Cross-Entropy

Common for single-label multiclass classification. It penalizes low probability assigned to the correct class. Depending on the library, targets may be class indices or one-hot vectors.

**Loss vs. metric:** loss is optimized during training; a metric such as accuracy, F1, or MAE is used to interpret performance. They can be related, but they are not interchangeable.

## Backpropagation and Gradients

**Backpropagation** efficiently computes how the loss changes with each trainable parameter by applying the chain rule from the output layer back through the network.

A gradient indicates the direction and rate of local change in loss with respect to a parameter. An optimizer uses gradients to update parameters.

For a parameter `w`, a basic gradient-descent update is:

`w_new = w_old - learning_rate * dL/dw`

Backpropagation calculates gradients; gradient descent or another optimizer uses those gradients to update parameters. They are related but different steps.

### Vanishing and Exploding Gradients

- **Vanishing gradients:** gradients become very small as they propagate through many layers or time steps, making early layers learn slowly.
- **Exploding gradients:** gradients become extremely large, causing unstable updates.
- **Mitigations:** suitable activations and initialization, residual connections, normalization, gated recurrent units, and gradient clipping where appropriate.

## Optimization Algorithms

| Optimizer | Core idea | Notes |
|---|---|---|
| Batch Gradient Descent | Uses the full training set for each update | Expensive for large datasets |
| Stochastic Gradient Descent (SGD) | Updates from one example at a time | Noisy updates |
| Mini-batch SGD | Updates from a small batch | Standard practical approach |
| Momentum | Accumulates a moving direction of past gradients | Can speed up consistent progress |
| RMSProp | Adapts updates using recent squared gradients | Useful for some non-stationary settings |
| Adam | Combines momentum-like and adaptive scaling | Strong baseline for many tasks |
| AdamW | Decouples weight decay from Adam's gradient update | Common modern default |

### Learning Rate

The learning rate controls the size of parameter updates.
- Too high: loss may oscillate or diverge.
- Too low: training may be very slow or stall.
- Learning-rate schedules reduce or otherwise adjust the rate during training.
- Warm-up gradually increases the learning rate at the start in some training setups.

Always monitor training and validation curves rather than assuming one learning rate is universally best.

## Training, Validation, and Test Sets

- **Training set:** used to fit model parameters.
- **Validation set:** used for model selection, hyperparameter tuning, and decisions such as early stopping.
- **Test set:** used for a final, relatively unbiased estimate after decisions are complete.

Do not repeatedly tune against the test set; doing so leaks information into the development process. For small datasets, cross-validation may help. Prevent leakage from duplicates, time ordering, users, or related records that could appear across splits.

### Epoch, Batch, and Iteration

- **Epoch:** one pass through the entire training set.
- **Batch:** a subset processed together.
- **Iteration / step:** usually one optimizer update.
- If there are 1,000 training examples and batch size 100, there are roughly 10 batches per epoch (depending on whether the final incomplete batch is kept).

## Overfitting and Regularization

- **Underfitting:** model is too limited or insufficiently trained; training and validation performance are both poor.
- **Overfitting:** model fits training examples well but performs worse on unseen data.
- **Generalization:** performance on unseen examples drawn from the intended data distribution.

Common methods to reduce overfitting:
- Collect more representative data or use valid augmentation.
- Reduce model capacity.
- Use weight decay / L2 regularization.
- Apply dropout.
- Use early stopping based on validation performance.
- Improve data quality and fix leakage.
- Use cross-validation when suitable.

### Dropout

During training, dropout randomly disables a fraction of activations to discourage fragile co-adaptations. It is normally disabled during inference, with framework-managed scaling.

### Weight Decay and L2

Both discourage excessively large weights, but implementation details differ by optimizer. In AdamW, weight decay is decoupled from the adaptive gradient update. Do not assume all uses of “L2” and “weight decay” are exactly equivalent.

### Data Augmentation

Creates valid variations of training examples, such as random crops or flips for suitable images. Augmentations must preserve the label and be realistic for the task. Never augment validation/test data in a way that contaminates the evaluation.

## Important Training Concepts

- **Weight initialization:** starts parameters at useful scales. Poor initialization can saturate activations or disrupt gradient flow.
- **Batch normalization:** normalizes activations using batch statistics during training and moving statistics during inference.
- **Layer normalization:** normalizes across features within an example; often useful in sequence and Transformer models.
- **Residual connection:** adds a layer's input to its transformed output, helping information and gradients flow through deeper networks.
- **Early stopping:** stops when validation performance no longer improves according to a chosen rule.
- **Class imbalance:** some labels are much more common than others. Accuracy alone can be misleading; inspect per-class precision, recall, F1, confusion matrix, and PR-AUC as appropriate.
- **Reproducibility:** record data versions, random seeds, package versions, preprocessing, hyperparameters, and checkpoints. Exact determinism may not be guaranteed across hardware or libraries.

## Common Neural Network Architectures

| Architecture | Best-known use | Main idea |
|---|---|---|
| Feedforward / MLP | Tabular data and basic classification/regression | Information flows from input to output |
| CNN | Images and spatial signals | Learns local patterns with shared filters |
| RNN | Sequential data | Reuses a hidden state over time |
| LSTM / GRU | Sequence tasks | Gating helps retain useful information |
| Transformer | Text, vision, audio, multimodal tasks | Attention relates elements in a sequence or input |
| Autoencoder | Compression, representation learning, anomaly detection | Learns to reconstruct inputs through a latent representation |
| GAN | Generative modeling | Generator and discriminator compete during training |
| Graph Neural Network | Graph-structured data | Aggregates information from connected nodes |

Choose architecture based on the data, task, latency, data volume, and available compute—not just popularity.

## Convolutional Neural Networks (CNNs)

CNNs are widely used for images and other grid-like data.

Core components:
- **Convolution:** applies learnable filters to local regions.
- **Kernel / filter:** a small matrix of learnable values.
- **Stride:** how far a filter moves at each step.
- **Padding:** adds border values, often zeros, to control spatial output size.
- **Feature map:** result produced by applying a filter.
- **Pooling:** downsamples spatial dimensions, such as max pooling.
- **Flattening / global pooling:** converts feature maps into features for a prediction head.

Early layers may learn edges and textures; deeper layers can combine them into more complex patterns. This is a useful intuition, not a guarantee that each layer has a human-readable meaning.

### CNN output-size intuition

For one spatial dimension, with input size `I`, kernel `K`, padding `P`, and stride `S`:

`O = floor((I + 2P - K) / S) + 1`

Exact behavior depends on the framework's padding convention and dilation.

## RNNs, LSTMs, and GRUs

RNNs process sequences step by step, updating a hidden state based on the current input and previous state. They are useful conceptually for text, speech, and time-series tasks, but basic RNNs can struggle with long-range dependencies.

- **LSTM:** uses a cell state and gates (input, forget, output) to control information flow.
- **GRU:** uses update and reset gates with a simpler structure than LSTM.
- **Limitation:** sequential processing can limit parallelism; gradients may vanish or explode over long sequences.
- **Modern context:** Transformers have replaced RNNs in many large-scale language tasks, but RNN variants can still be useful in constrained or streaming applications.

## Transformers and Attention

Transformers use attention to combine information from different input positions. They can process many positions in parallel during training, although autoregressive generation still produces tokens sequentially in common setups.

### Self-attention intuition

Each token is projected into:
- **Query (Q):** what this position is looking for.
- **Key (K):** what information this position offers.
- **Value (V):** the content to combine.

Scaled dot-product attention is commonly written:

`Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`

- **Multi-head attention:** learns multiple attention patterns in parallel.
- **Positional information:** provides order information because attention alone does not inherently encode sequence order.
- **Encoder:** builds contextual representations of inputs.
- **Decoder:** generates outputs and, in autoregressive models, uses causal masking to prevent attending to future tokens.
- **Causal mask:** blocks future positions during next-token prediction.
- **Tokenization:** converts text into token IDs.
- **Embedding:** maps token IDs to learned vectors.
- **Context window:** maximum sequence context a model can process in one pass, subject to the model/system design.

### LLM basics

Large language models (LLMs) are commonly Transformer-based models trained on large text datasets, often with next-token prediction as a pretraining objective. Additional instruction tuning and preference/alignment methods can make them more useful for following requests. LLMs can still produce incorrect claims, so factual and high-impact outputs need appropriate verification.

## Autoencoders and GANs

### Autoencoder

An autoencoder has:
- **Encoder:** maps the input to a compressed latent representation.
- **Latent space:** compact internal representation.
- **Decoder:** reconstructs the input from that representation.
- **Reconstruction loss:** measures the difference between input and reconstruction.

Uses include representation learning, denoising, dimensionality reduction, and some anomaly-detection systems. Reconstruction error is not automatically a reliable anomaly score; it must be validated on realistic data.

### Generative Adversarial Network (GAN)

A GAN contains:
- **Generator:** creates synthetic samples.
- **Discriminator:** tries to distinguish real samples from generated ones.

The networks train in opposition. GANs can generate realistic outputs, but training can be unstable and may suffer from mode collapse, where the generator produces insufficient variety.

## Transfer Learning and Fine-Tuning

**Transfer learning** reuses representations learned on one task or dataset for another.

Typical workflow:
1. Start from a pretrained model.
2. Replace or adapt its task-specific output head.
3. Freeze most of the backbone and train the new head as a baseline.
4. Optionally unfreeze selected layers and fine-tune with a smaller learning rate.
5. Compare against a baseline on a clean validation set.

**Feature extraction:** keep pretrained layers frozen and use their outputs as features.

**Fine-tuning:** update some or all pretrained parameters on the target task.

Benefits include less data and compute than training from scratch. Risks include domain mismatch, catastrophic forgetting, and overfitting on small target datasets.

## Evaluation Metrics

### Classification

- **Accuracy:** fraction of predictions that are correct.
- **Precision:** among predicted positives, fraction that are truly positive.
- **Recall / sensitivity:** among actual positives, fraction found.
- **Specificity:** among actual negatives, fraction correctly identified.
- **F1 score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts of true/false positives/negatives.
- **ROC-AUC:** measures ranking across thresholds using true-positive and false-positive rates.
- **PR-AUC:** often informative for rare positive classes; interpretation depends on prevalence.

Formulas:

`Precision = TP / (TP + FP)`

`Recall = TP / (TP + FN)`

`F1 = 2 * Precision * Recall / (Precision + Recall)`

### Regression

- **MAE:** mean absolute error; interpretable in target units.
- **MSE:** mean squared error; penalizes large errors more strongly.
- **RMSE:** square root of MSE; in target units.
- **R-squared:** compares fit to a baseline based on the target mean; can be negative on held-out data.

### Choosing a metric

Choose metrics that reflect the real cost of mistakes. For rare disease screening or fraud detection, missing a positive case may be more costly than investigating a false alarm, so recall and precision may matter more than accuracy. Always establish a simple baseline.

## Practical Workflow

1. **Define the task:** inputs, targets, intended users, and success metric.
2. **Inspect the data:** missing values, duplicates, class balance, data quality, leakage, and label quality.
3. **Split correctly:** use stratification for suitable classification tasks and time/group-aware splits where needed.
4. **Build a baseline:** compare with a simple model before increasing complexity.
5. **Preprocess:** normalize or standardize numeric inputs when appropriate; fit preprocessing only on training data.
6. **Train:** select architecture, loss, optimizer, batch size, and learning rate.
7. **Monitor:** track training/validation loss and metrics.
8. **Diagnose:** underfitting, overfitting, unstable gradients, bad labels, or pipeline errors.
9. **Tune:** change a small number of hyperparameters systematically.
10. **Evaluate once on the test set:** after model choices are complete.
11. **Deploy responsibly:** consider latency, cost, privacy, security, drift, monitoring, and rollback.
12. **Document:** dataset provenance, limitations, metrics, reproducibility steps, and ethical considerations.

## Python Example: A Tiny Neural Network

This example uses PyTorch to classify synthetic two-dimensional data. It is a teaching example, not a production model.

```python
# Install PyTorch using the instructions at https://pytorch.org/get-started/locally/
import torch
from torch import nn

torch.manual_seed(42)

# Two-dimensional inputs and binary labels
X = torch.randn(1000, 2)
y = ((X[:, 0] * X[:, 1]) > 0).long()

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)  # Two class logits
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    logits = model(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()       # Backpropagation computes gradients
    optimizer.step()      # Optimizer updates parameters

    if (epoch + 1) % 20 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
        print(f"Epoch {epoch + 1:3d} | loss={loss.item():.4f} | accuracy={accuracy:.3f}")
```

What to notice:
- `nn.Linear` defines trainable dense layers.
- `ReLU` adds non-linearity.
- `CrossEntropyLoss` expects raw logits and integer class labels in this setup.
- `zero_grad()` clears gradients from the previous update.
- `backward()` computes gradients.
- `step()` changes parameters.
- This demonstration does not include a separate validation/test set; real experiments should.

## Deep Learning Interview FAQ

### Fundamentals

**1. What is deep learning?**  
Deep learning is a machine-learning approach that learns representations using neural networks with multiple layers. It is especially effective for complex data such as images, audio, and text when enough suitable data and compute are available.

**2. How is deep learning different from traditional machine learning?**  
Deep learning often learns features and predictions jointly through multi-layer networks. Traditional ML often relies more on manually designed features and can be easier, faster, and more effective for smaller structured datasets.

**3. What is an artificial neuron?**  
A unit that computes a weighted sum of inputs plus a bias and applies an activation function.

**4. Why do neural networks need activation functions?**  
They introduce non-linearity. Without non-linear activations, a stack of linear layers remains equivalent to a single linear transformation.

**5. What are weights and biases?**  
Weights determine how strongly inputs contribute; biases shift a neuron's pre-activation. Training learns these parameters.

**6. What is the difference between a parameter and a hyperparameter?**  
Parameters are learned from data, such as weights. Hyperparameters are chosen for training, such as learning rate, batch size, and layer count.

**7. What is an epoch?**  
One complete pass through the training dataset.

**8. What is batch size?**  
The number of examples processed in one training batch before an optimizer update.

**9. What is a learning rate?**  
It controls the scale of optimizer updates. Too high can destabilize training; too low can make learning slow.

**10. What is a loss function?**  
A function that quantifies prediction error and provides the objective minimized during training.

### Training and Optimization

**11. Explain forward propagation.**  
Inputs pass through network layers to produce predictions. The loss is then calculated from predictions and targets.

**12. Explain backpropagation.**  
It applies the chain rule to calculate the loss gradient with respect to parameters, propagating gradient information from the output toward earlier layers.

**13. Are backpropagation and gradient descent the same?**  
No. Backpropagation calculates gradients; gradient descent or another optimizer uses them to update parameters.

**14. What is gradient descent?**  
An optimization method that updates parameters in the direction opposite the loss gradient.

**15. What is the difference between batch, stochastic, and mini-batch gradient descent?**  
Batch uses all training examples per update, stochastic uses one example, and mini-batch uses a subset. Mini-batch training is common in practice.

**16. What is Adam?**  
An adaptive optimizer that uses moving estimates of gradients and squared gradients to scale updates. AdamW is a common variant with decoupled weight decay.

**17. What is the vanishing-gradient problem?**  
Gradients become very small during backpropagation, making early layers learn slowly. It can occur in deep networks and long recurrent sequences.

**18. What is the exploding-gradient problem?**  
Gradients grow excessively large, causing unstable parameter updates. Gradient clipping and better architecture/initialization may help.

**19. Why is weight initialization important?**  
It helps keep activations and gradients at useful scales at the beginning of training. Poor initialization can slow or destabilize learning.

**20. What is a learning-rate scheduler?**  
A strategy that adjusts the learning rate over training, for example reducing it when progress plateaus.

### Generalization

**21. What is overfitting?**  
The model learns training-specific patterns or noise and performs worse on unseen data.

**22. What is underfitting?**  
The model fails to learn the training patterns adequately, often because it is too simple, poorly configured, or insufficiently trained.

**23. How can overfitting be reduced?**  
Use more representative data, valid augmentation, dropout, weight decay, early stopping, a smaller model, or improved data quality.

**24. What is dropout?**  
A regularization method that randomly disables some activations during training. It is usually turned off during inference.

**25. Why do we need validation data?**  
To compare models, tune hyperparameters, and make training decisions without using the final test set.

**26. What is data leakage?**  
Information unavailable at prediction time—or information from the validation/test set—enters training or preprocessing, making evaluation misleadingly optimistic.

**27. Why can accuracy be misleading on imbalanced data?**  
A model can predict only the majority class and still achieve high accuracy while failing on the minority class. Use per-class metrics and a confusion matrix.

### Architectures

**28. What is a CNN?**  
A neural network using convolutional filters to learn local spatial patterns. CNNs are widely used in image tasks.

**29. What are convolution, stride, and padding?**  
Convolution applies a filter; stride controls its movement; padding adds border values to influence output dimensions.

**30. What is pooling?**  
A downsampling operation that reduces spatial dimensions, such as max pooling. Some modern architectures instead use strided convolutions or global pooling.

**31. What is an RNN?**  
A network that processes a sequence while maintaining a hidden state across time steps.

**32. How do LSTM and GRU improve on a basic RNN?**  
Their gating mechanisms help control information flow and can better preserve useful information over longer sequences.

**33. What is attention?**  
A mechanism that weights information from different input positions according to their relevance to a query.

**34. What is a Transformer?**  
A neural-network architecture built around attention and feed-forward layers, widely used in language and increasingly in vision, audio, and multimodal systems.

**35. What is self-attention?**  
Attention in which positions in a sequence use representations from the same sequence to compute how information should be combined.

**36. What are query, key, and value in attention?**  
Learned projections of token representations: queries select relevant information, keys determine compatibility, and values carry the information combined by attention.

**37. Why do Transformers need positional information?**  
Attention alone does not inherently represent token order. Positional encodings or related mechanisms provide information about position.

**38. What is transfer learning?**  
Reusing a model or representations learned on one task for another related task.

**39. What is fine-tuning?**  
Continuing training of a pretrained model on target-task data, usually with carefully chosen learning rates and validation.

**40. What is an autoencoder?**  
A network trained to encode input into a latent representation and reconstruct the input.

**41. What is a GAN?**  
A generative framework in which a generator creates samples and a discriminator learns to distinguish generated from real samples.

### Evaluation and Practical Work

**42. What is the difference between classification and regression?**  
Classification predicts discrete classes; regression predicts numeric values.

**43. When would you use binary cross-entropy?**  
For binary classification or independent binary labels, using the appropriate sigmoid/logits formulation for the framework.

**44. When would you use cross-entropy?**  
For classification tasks where the target is a class or class distribution; the exact variant depends on whether classes are mutually exclusive or labels are independent.

**45. Precision vs. recall?**  
Precision asks, “Of the predicted positives, how many are correct?” Recall asks, “Of the actual positives, how many did we find?”

**46. What is F1 score?**  
The harmonic mean of precision and recall, useful when both matter. It depends on the chosen classification threshold.

**47. What is a confusion matrix?**  
A table comparing true labels with predicted labels, showing correct and incorrect predictions by class.

**48. How would you debug a model whose loss does not decrease?**  
Check labels and tensor shapes, preprocessing, loss/output compatibility, gradient flow, learning rate, parameter updates, and whether a tiny batch can be overfit as a sanity check.

**49. How do you choose a model architecture?**  
Start from the task, data type, dataset size, latency and compute limits, and a simple baseline. Add complexity only when validation results justify it.

**50. What should be included in a deep learning project README?**  
Problem statement, dataset and license/provenance, setup, preprocessing, model architecture, training command, metrics, reproducibility details, limitations, and examples of use.

**51. What is inference?**  
Using a trained model to make predictions on new inputs, typically without updating model parameters.

**52. What is a checkpoint?**  
A saved model state (often including optimizer and scheduler state) used to resume training or deploy a chosen model.

**53. What is model drift?**  
A change in data patterns or the relationship between inputs and outcomes that can degrade production performance. Monitor data and model metrics and retrain when justified.

**54. What is explainability in deep learning?**  
Methods that help people understand model behavior, such as feature attribution or saliency maps. These explanations have limitations and should not be treated as perfect causal accounts.

**55. What are ethical concerns in deep learning?**  
Bias, privacy, consent, security, explainability, environmental cost, misuse, and unequal error rates. Evaluate the model in the context where it will be used.

## Practice Questions

Try answering without looking at the notes:

1. Draw a neural network and explain a forward pass.
2. Explain the chain rule using a two-layer network.
3. Compare sigmoid, tanh, ReLU, and softmax.
4. Explain why logits are often passed directly to a cross-entropy loss.
5. Diagnose a training curve where training loss falls but validation loss rises.
6. Choose metrics for an imbalanced fraud-detection problem and justify them.
7. Compare CNNs and Transformers for image tasks.
8. Explain how an LSTM gate works at a high level.
9. Design a leakage-safe train/validation/test split for time-series data.
10. Build a small classifier and report precision, recall, F1, and a confusion matrix.
11. Explain feature extraction vs. fine-tuning.
12. Describe how you would deploy and monitor a model responsibly.

## Glossary

- **Activation:** output of a neuron after applying its activation function.
- **Backpropagation:** gradient calculation using the chain rule.
- **Batch:** examples processed together in one step.
- **Checkpoint:** saved model state.
- **Epoch:** one pass through training data.
- **Embedding:** vector representation of a discrete item such as a token.
- **Generalization:** ability to perform on unseen data.
- **Gradient:** derivative of loss with respect to a parameter.
- **Hyperparameter:** user-selected training/model setting.
- **Inference:** prediction using a trained model.
- **Latent representation:** learned internal encoding.
- **Logits:** unnormalized model scores before a probability transformation.
- **Regularization:** methods that discourage poor generalization.
- **Tensor:** multidimensional array used by deep-learning libraries.
- **Token:** unit produced by a tokenizer, such as a word fragment or symbol.

## Learning Resources

Use the official and well-established resources below; check their current versions and installation instructions.

- **PyTorch tutorials:** https://pytorch.org/tutorials/
- **TensorFlow tutorials:** https://www.tensorflow.org/tutorials
- **Dive into Deep Learning (free online book):** https://d2l.ai/
- **Deep Learning book by Goodfellow, Bengio, and Courville:** https://www.deeplearningbook.org/
- **3Blue1Brown — Neural Networks:** https://www.3blue1brown.com/topics/neural-networks
- **scikit-learn user guide (preprocessing and evaluation foundations):** https://scikit-learn.org/stable/user_guide.html

## Suggested Study Order

1. Neurons, dense layers, activations, and forward propagation.
2. Loss functions, gradients, backpropagation, and optimizers.
3. Train/validation/test splits, metrics, and overfitting.
4. Build a small MLP in PyTorch.
5. Study CNNs and implement an image classifier.
6. Learn sequence modeling, attention, and Transformers.
7. Try transfer learning on a small real dataset.
8. Document experiments and practice the FAQ aloud.

---

**Revision tip:** For each topic, be ready to give (1) a one-sentence definition, (2) the intuition, (3) one use case, and (4) one limitation.
