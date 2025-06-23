### Agent
Agent: an AI model capable of reasoning, planning, and interacting with its environment.
We call it Agent because it has agency, aka it has the ability to interact with the environment.

##### An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks.

an Agent is a system that uses an AI Model (typically an LLM) as its core reasoning engine, to:

Understand natural language: Interpret and respond to human instructions 
Reason and plan: Analyze information, make decisions, and devise strategies to solve problems.

Interact with its environment: Gather information, take actions, and observe the results of those actions.

Actions are the steps the Agent takes, while Tools are external resources the Agent can use to perform those actions.

LLMs serve as the reasoning 'brain' of the Agent, processing text inputs to understand instructions and plan actions.

There are 3 types of transformers:

1. Encoders
An encoder-based Transformer takes text (or other data) as input and outputs a dense representation (or embedding) of that text.

Example: BERT from Google
Use Cases: Text classification, semantic search, Named Entity Recognition
Typical Size: Millions of parameters

2. Decoders
A decoder-based Transformer focuses on generating new tokens to complete a sequence, one token at a time.

Example: Llama from Meta
Use Cases: Text generation, chatbots, code generation
Typical Size: Billions (in the US sense, i.e., 10^9) of parameters

3. Seq2Seq (Encoder–Decoder)
A sequence-to-sequence Transformer combines an encoder and a decoder. The encoder first processes the input sequence into a context representation, then the decoder generates an output sequence.

Example: T5, BART
Use Cases: Translation, Summarization, Paraphrasing
Typical Size: Millions of parameters

Although Large Language Models come in various forms, LLMs are typically decoder-based models with billions of parameters.

LLMs are said to be autoregressive, meaning that the output from one pass becomes the input for the next one. This loop continues until the model predicts the next token to be the EOS token, at which point the model can stop.

Once the input text is tokenized, the model computes a representation of the sequence that captures information about the meaning and the position of each token in the input sequence.

This representation goes into the model, which outputs scores that rank the likelihood of each token in its vocabulary as being the next one in the sequence.

Beam search explores multiple candidate sequences to find the one with the maximum total score–even if some individual tokens have lower scores.

Parameters:
Sentence to decode from (inputs): the input sequence to your decoder.
Number of steps (max_new_tokens): the number of tokens to generate.
Number of beams (num_beams): the number of beams to use.
Length penalty (length_penalty): the length penalty to apply to outputs. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences. This parameter will not impact the beam search paths, but only influence the choice of sequences in the end towards longer or shorter sequences.
Number of return sequences (num_return_sequences): the number of sequences to be returned at the end of generation. Should be <= num_beams.

context length, which refers to the maximum number of tokens the LLM can process, and the maximum attention span it has.

