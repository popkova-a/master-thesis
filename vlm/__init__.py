"""
Fifth Experiment:

The fifth experiment focused on combining the GET model (event encoder) with DeepSeek-VL’s
aligner and LLM, leveraging the pretrained image-conditioned alignment which proves
significantly simpler than direct event-to-language alignment. We prepended event embeddings
as soft visual prompts to text queries, to condition the LLM on the event-based visual representations.
Finetuning just the encoder and aligner on a bootstrapped classification CIFAR10DVS dataset yielded
English outputs (excluding digits/Chinese characters from pretraining), but remained meaningless –
showing the necessity for specialized event-language alignment beyond basic finetuning.

------------------------------------------------------------------------------------------------------
Sixth Experiment:

Event simulation from a video dataset using v2e.

------------------------------------------------------------------------------------------------------
Seventh Experiment:

In the seventh experiment, we pretrained the event encoder on event sequences from CIFAR10-DVS using
spatiotemporal augmentations such as horizontal flip, polarity flip, EventDrop, and temporal clipping.
We employed a SimCLR-style contrastive learning objective with a batch size of 1024, a temperature (τ)
of 0.1, and trained for 200 epochs. Increasing the batch size improved the accuracy of a linear probe
evaluated on frozen features from the CIFAR10-DVS validation set.

Further fine-tuning of the pretrained encoder with a classification head achieved 78.5% accuracy.
Considering the model's compactness (4.5M parameters) compared to other event-based pretrained models
that achieve ~79% accuracy, this result is highly promising. Visualization of t-SNE embeddings demonstrated
that pretraining effectively leveraged spatiotemporal characteristics of the data to improve class
separation in the downstream task.

Additionally, we redesigned the event tokenizer to handle input from arbitrary sensor sizes by dynamically
resizing events to a reference resolution.

In summary, our SimCLR-based self-supervised pretraining strategy successfully equips the model with robust
spatiotemporal understanding, which benefits downstream tasks. Unlike other methods, our augmentations
explicitly preserve the spatiotemporal properties of event data, ensuring the model learns meaningful
representations.

We integrated the pretrained event encoder with DeepSeek-LLM using a two-stage training procedure. In the
first stage, the encoder remained frozen while we trained a multilayer perceptron (MLP) aligner. This phase
plateaued at a loss value comparable to training the encoder and aligner from scratch, suggesting limited
initial alignment between modalities. Notably, the model exhibited hallucinations in validation examples—though
output text contained relevant keywords, it degenerated into repetitive semantic fragments (e.g., for a cat event
sequence, the output repeatedly generated "...independent cat.. and cat.. and cat. cat. cat. cat...").

During the second stage, we unfroze the encoder and jointly optimized both components. The training loss
decreased monotonically, but the validation loss oscillated persistently, indicating overfitting. This
divergence implies that the model’s capacity—coupled with limited paired event-text data—led to memorization
rather than generalized cross-modal alignment.

The SimCLR-based event encoder captures useful spatiotemporal features, but deeper alignment between visual
and textual representations is necessary. Bootstrapping classification datasets via class-description extensions
proved insufficient for robust alignment. The observed overfitting underscores the necessity of large-scale,
high-quality event-text paired datasets. While the encoder’s pretraining is sound, bridging event-data and
language domains may require either (a) more sophisticated alignment mechanisms (e.g., cross-attention) or
(b) significantly larger multimodal pretraining corpora.

------------------------------------------------------------------------------------------------------
Eighth Experiment:

SimCLR for larger encoder + CLIP event-text alignment + finetuning on the resulting event-text paired dataset.
"""