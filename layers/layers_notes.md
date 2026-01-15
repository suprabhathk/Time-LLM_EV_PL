---
title:  Notes about how the layers folder works in this repository
---

### Layers folder

#### This contains an inisght into how the layers folder functions in the context of TimeLLM paper forecasting

##### 1. StandardNorm
* This is basically done to standardize the input time series (X-raw -> X-norm) by setting the mean to zero and variance to one. For stabilizing training and enables later de-standardization
* Uses something called ***reversible instance normalization (RevIN)*** [RevIN paper](https://openreview.net/pdf?id=cGDAkQo1C0p)
* Structure of the code in this python file
  * Normalize class: 
    * Initializes the paramters and settings for the normalization layer
    * *forward* function : applies normalization or renormalization based on the mode (Why is it needed? Investigate)
    * *init_params*
      * contains code to initialize the RevIN parameters
        * needs affine_weight and affine_bias
      * *_get_statistics* to calculate mean and std dev
      * *_normalize* and *denormalize*

##### 2. Embed
* Once the normalization/standardization of the input time series is done - the file referenced in this section proceeds towards embedding the input data to latent feature spaces for the main body of the network.
* ***PositionalEmbedding*** : Assigns sine/cosine aspects to the embedding to give the transformers some info about the position of each text - computes positional encodings in the log space
* ***TokenEmbedding*** : Initial transformation of numerical time series data into high dimensional vectors (embeddings) that can be processed by subsequent transformal layer. Its done because of two reasons : 
  * Dimension matching : to have consisten dimensions as the dimensions of the model we are using
  * Feature abstraction : transforms raw, concrete numerical values into abstract semantic features
* ***FixedEmbedding*** : To inject explicity, periodic time context into the data stream - in the code its the same as ***PositionalEmbedding***. Not sure why it was created
* ***TemporalEmbedding*** : Basically converting input into a time based embedding vector
* ***DataEmbedding*** : Combine all above embeddings into one representation (with and without positional embeddings and without temporal embeddings)
* ***ReplicationPad1d*** : Ensures the input time series sequence length is compatible with specified patch isze and stride by adding padding to the end of the sequence
* ***PatchEmbedding*** : Primary input layer responsible for slicing time series into overlapping patches, embedding the feature vector of each patch and applying necesary regularization

##### 3. Autoformer_EncDec

##### 4. SelfAttention_Family

##### 5. AutoCorrelation

* It is done to basically to handle temporal dependency - by identifying periodicity (seasonal or periodic patterns) and to do some feature extraction - which essentiall use autocorrelation
* Standard transformer architectures use self-attention to calculate dependencies - leading to computational complexity dependent on input length
* Execution in the paper :
  * Contain the hidden representation of the time series (X)
  * X is lineearly transformed to create the Q,K,V tensors
  * Q and K are transofrmed using Fast Fourier Transform (FFT)
  * Autocorrelation R between Q and K is computed as an Inverse Fourier Transform (IFT) of the product of their frequency representations.
  * Top-k significant periods corresponding to the highest autocorrelation scores are selected.
  * Instead of calculating a full attention matrix - model uses top-k periods to roll and aggregate information from the V tensor - using Weight Calculation (uses Softmax) and Time Delay rolling (Roll meaning shifted). Final output is weighted sum.
  * Followed by the traditional transformer architecture.

##### 6. Conv_Blocks

##### 7. Transformer_EncDec
* 
