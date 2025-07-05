## What is an Agent?
Agent is an AI model capable of reasoning, planning, and interacting with its environment.
We call it Agent because it has agency, aka it has the ability to interact with the environment.

##### An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks.

#### AI agents are programs that use LLMs to generate ‘thoughts’ based on ‘observations’ to perform ‘actions’.

an Agent is a system that uses an AI Model (typically an LLM) as its core reasoning engine, to:

##### Understand natural language: Interpret and respond to human instructions 
##### Reason and plan: Analyze information, make decisions, and devise strategies to solve problems.
##### Interact with its environment: Gather information, take actions, and observe the results of those actions.

##### Actions are the steps the Agent takes, while Tools are external resources the Agent can use to perform those actions.

#### LLM: A deep learning model trained on large amounts of text to understand and generate human-like language

LLMs serve as the reasoning 'brain' of the Agent, processing text inputs to understand instructions and plan actions.

### Transformers

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

### Beam Search
Beam search explores multiple candidate sequences to find the one with the maximum total score–even if some individual tokens have lower scores.

##### Parameters:
Sentence to decode from (inputs): the input sequence to your decoder.
Number of steps (max_new_tokens): the number of tokens to generate.
Number of beams (num_beams): the number of beams to use.
Length penalty (length_penalty): the length penalty to apply to outputs. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences. This parameter will not impact the beam search paths, but only influence the choice of sequences in the end towards longer or shorter sequences.
Number of return sequences (num_return_sequences): the number of sequences to be returned at the end of generation. Should be <= num_beams.

context length, which refers to the maximum number of tokens the LLM can process, and the maximum attention span it has.

chat templates are essential for structuring conversations between language models and users. They guide how message exchanges are formatted into a single prompt.

##### A Base Model is trained on raw text data to predict the next token.

##### An Instruct Model is fine-tuned specifically to follow instructions and engage in conversations. For example, SmolLM2-135M is a base model, while SmolLM2-135M-Instruct is its instruction-tuned variant.

To make a Base Model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in.

In transformers, chat templates include Jinja2 code that describes how to transform the ChatML list of JSON messages

A Tool should contain:

A textual description of what the function does.
A Callable (something to perform an action).
Arguments with typings.
(Optional) Outputs with typings.

This textual description is what we want the LLM to know about the tool.
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int

##### Auto-formatting Tool sections
@tool

##### Generic Tool implementation

Model Context Protocol (MCP): a unified tool interface
Model Context Protocol (MCP) is an open protocol that standardizes how applications provide tools to LLMs. MCP provides:

A growing list of pre-built integrations that your LLM can directly plug into
The flexibility to switch between LLM providers and vendors
Best practices for securing your data within your infrastructure
This means that any framework implementing MCP can leverage tools defined within the protocol, eliminating the need to reimplement the same tool interface for each framework.

### The Core Components
Agents work in a continuous cycle of: thinking (Thought) → acting (Act) and observing (Observe).

Let’s break down these actions together:

##### Thought: The LLM part of the Agent decides what the next step should be.
##### Action: The agent takes an action, by calling the tools with the associated arguments.
##### Observation: The model reflects on the response from the tool.

What we see in this example:

Agents iterate through a loop until the objective is fulfilled:
Alfred’s process is cyclical. It starts with a thought, then acts by calling a tool, and finally observes the outcome. If the observation had indicated an error or incomplete data, Alfred could have re-entered the cycle to correct its approach.

##### Tool Integration:
The ability to call a tool (like a weather API) enables Alfred to go beyond static knowledge and retrieve real-time data, an essential aspect of many AI Agents.

##### Dynamic Adaptation:
Each cycle allows the agent to incorporate fresh information (observations) into its reasoning (thought), ensuring that the final answer is well-informed and accurate.

### The ReAct Approach
A key method is the ReAct approach, which is the concatenation of “Reasoning” (Think) with “Acting” (Act).

ReAct is a simple prompting technique that appends “Let’s think step by step” before letting the LLM decode the next tokens.

Indeed, prompting the model to think “step by step” encourages the decoding process toward next tokens that generate a plan, rather than a final solution, since the model is encouraged to decompose the problem into sub-tasks.

This allows the model to consider sub-steps in more detail, which in general leads to less errors than trying to generate the final solution directly.

### The Stop and Parse Approach
One key method for implementing actions is the stop and parse approach. This method ensures that the agent’s output is structured and predictable:

##### Generation in a Structured Format:
The agent outputs its intended action in a clear, predetermined format (JSON or code).

##### Halting Further Generation:
Once the text defining the action has been emitted, the LLM stops generating additional tokens. This prevents extra or erroneous output.

##### Parsing the Output:
An external parser reads the formatted action, determines which Tool to call, and extracts the required parameters. 

### Code Agents
An alternative approach is using Code Agents. The idea is: instead of outputting a simple JSON object, a Code Agent generates an executable code block—typically in a high-level language like Python.

This approach offers several advantages:

- Expressiveness: Code can naturally represent complex logic, including loops, conditionals, and nested functions, providing greater flexibility than JSON.
- Modularity and Reusability: Generated code can include functions and modules that are reusable across different actions or tasks.
- Enhanced Debuggability: With a well-defined programming syntax, code errors are often easier to detect and correct.
- Direct Integration: Code Agents can integrate directly with external libraries and APIs, enabling more complex operations such as data processing or real-time decision making.
You must keep in mind that executing LLM-generated code may pose security risks, from prompt injection to the execution of harmful code. That’s why it’s recommended to use AI agent frameworks like smolagents that integrate default safeguards.

Malicious code execution can occur in several ways:

- Plain LLM error: LLMs are still far from perfect and may unintentionally generate harmful commands while attempting to be helpful. While this risk is low, instances have been observed where an LLM attempted to execute potentially dangerous code.
- Supply chain attack: Running an untrusted or compromised LLM could expose a system to harmful code generation. While this risk is extremely low when using well-known models on secure inference infrastructure, it remains a theoretical possibility.
- Prompt injection: an agent browsing the web could arrive on a malicious website that contains harmful instructions, thus injecting an attack into the agent’s memory
- Exploitation of publicly accessible agents: Agents exposed to the public can be misused by malicious actors to execute harmful code. Attackers may craft adversarial inputs to exploit the agent’s execution capabilities, leading to unintended consequences. Once malicious code is executed, whether accidentally or intentionally, it can damage the file system, exploit local or cloud-based resources, abuse API services, and even compromise network security.

Observations are how an Agent perceives the consequences of its actions.

They provide crucial information that fuels the Agent’s thought process and guides future actions.

They are signals from the environment—whether it’s data from an API, error messages, or system logs—that guide the next cycle of thought.

In the observation phase, the agent:

- Collects Feedback: Receives data or confirmation that its action was successful (or not).
- Appends Results: Integrates the new information into its existing context, effectively updating its memory.
- Adapts its Strategy: Uses this updated context to refine subsequent thoughts and actions.

For example, if a weather API returns the data “partly cloudy, 15°C, 60% humidity”, 
##### this observation is appended to the agent’s memory (at the end of the prompt).

The Agent then uses it to decide whether additional information is needed or if it’s ready to provide a final answer.

This iterative incorporation of feedback ensures the agent remains dynamically aligned with its goals, constantly learning and adjusting based on real-world outcomes.

These observations can take many forms, from reading webpage text to monitoring a robot arm’s position. This can be seen like Tool “logs” that provide textual feedback of the Action execution.

##### How Are the Results Appended?
After performing an action, the framework follows these steps in order:

- Parse the action to identify the function(s) to call and the argument(s) to use.
- Execute the action.
- Append the result as an Observation.

### smolagents 

a library that focuses on codeAgent, a kind of agent that performs “Actions” through code blocks, and then “Observes” results by executing the code.

We will use Qwen/Qwen2.5-Coder-32B-Instruct as the LLM engine. This is a very capable model that we’ll access via the serverless API.

"Serverless API": Instead of running the model on your own server, you'll access it through a cloud-based API — the infrastructure is managed for you. You just send a request and get a response, without worrying about deploying or maintaining the model.


## When to Use an Agentic Framework
- An LLM engine that powers the system.
- A list of tools the agent can access.
- A parser for extracting tool calls from the LLM output.
- A system prompt synced with the parser.
- A memory system.
- Error logging and retry mechanisms to control LLM mistakes.

smolagents: Agents framework developed by Hugging Face.	
Llama-Index: End-to-end tooling to ship a context-augmented AI agent to production
LangGraph: Agents allowing stateful orchestration of agents

- code agents designed for software development tasks, 
- tool calling agents for creating modular, 
- function-driven workflows, and 
- retrieval agents that access and synthesize information

+ orchestration of multiple agents 
+ the integration of vision capabilities and web browsing

### smolagents
#### CodeAgents are the primary type of agent in smolagents. Instead of generating JSON or text, these agents produce Python code to perform actions. 

#### ToolCallingAgents are the second type of agent supported by smolagents. Unlike CodeAgents, which generate Python code, these agents rely on JSON/text blobs that the system must parse and interpret to execute actions. 

#### Retrieval agents allow models access to knowledge bases, making it possible to search, synthesize, and retrieve information from multiple sources. They leverage vector stores for efficient retrieval and implement Retrieval-Augmented Generation (RAG) patterns. These agents are particularly useful for integrating web search with custom knowledge bases while maintaining conversation context through memory systems.

#### Orchestrating multiple agents effectively is crucial for building powerful, multi-agent systems. By combining agents with different capabilities—such as a web search agent with a code execution agent—you can create more sophisticated solutions.

#### Vision agents extend traditional agent capabilities by incorporating Vision-Language Models (VLMs), enabling them to process and interpret visual information. 

### Key Advantages of smolagents
- Simplicity: Minimal code complexity and abstractions, to make the framework easy to understand, adopt and extend
- Flexible LLM Support: Works with any LLM through integration with Hugging Face tools and external APIs
- Code-First Approach: First-class support for Code Agents that write their actions directly in code, removing the need for parsing and simplifying tool calling
- HF Hub Integration: Seamless integration with the Hugging Face Hub, allowing the use of Gradio Spaces as tools

### When to use smolagents?
- You need a lightweight and minimal solution.
- You want to experiment quickly without complex configurations.
- Your application logic is straightforward.

### Code vs. JSON Actions
Unlike other frameworks where agents write actions in JSON, smolagents focuses on tool calls in code, simplifying the execution process. This is because there’s no need to parse the JSON in order to build code that calls the tools: the output can be executed directly.

![the difference between Code vs JSON Actions](https://cdn-lfs.hf.co/datasets/huggingface/documentation-images/2d3c40c6213af765c3caff5a18210cd75f5722ce6a012f99a5eb4cb6536965fc?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27code_vs_json_actions.png%3B+filename%3D%22code_vs_json_actions.png%22%3B&response-content-type=image%2Fpng&Expires=1751663257&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MTY2MzI1N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy8yZDNjNDBjNjIxM2FmNzY1YzNjYWZmNWExODIxMGNkNzVmNTcyMmNlNmEwMTJmOTlhNWViNGNiNjUzNjk2NWZjP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=Wmtt2hnmoM3p4qUW1tEJq6IJhw07uyKVFtAXKZHLsO2BdYQV%7EM3rDxo-CVtCpvmxDx0zkLnPGyOyfWQrX0FOIuDmTICQeUMg72hvNVaiEe5QcyVL2NDDW5stDEpjsT-X0QWpIIub9wJtoClp7zl7JihfVu7tSWq9dR-OMHXbIja9tFJTwWNFDNC70EJaMyTWlJLEidoeRWHNjZOAMWf4umALVUdRPpYXss%7EM96XqTP3sejq7BU0bz5sc5atF-9-7QXldSSGUG1IQdPFhJHjfn924f4RQfQwG5xdZNOVb4GMLbV4axWozzJue0yYN3aS2cv11PoHDPST3MctpsikhXA__&Key-Pair-Id=K3RPWS32NSSJCE)

The framework provides several predefined classes to simplify model connections:

- TransformersModel: Implements a local transformers pipeline for seamless integration.
- InferenceClientModel: Supports serverless inference calls through Hugging Face’s infrastructure, or via a growing number of third-party inference providers.
- LiteLLMModel: Leverages LiteLLM for lightweight model interactions.
- OpenAIServerModel: Connects to any service that offers an OpenAI API interface.
- AzureOpenAIServerModel: Supports integration with any Azure OpenAI deployment.

### Why Code Agents?
In a multi-step agent process, the LLM writes and executes actions, typically involving external tool calls. Traditional approaches use a JSON format to specify tool names and arguments as strings, which the system must parse to determine which tool to execute. However, research shows that tool-calling LLMs work more effectively with code directly.

Writing actions in code rather than JSON offers several key advantages:

- Composability: Easily combine and reuse actions
- Object Management: Work directly with complex structures like images
- Generality: Express any computationally possible task
- Natural for LLMs: High-quality code is already present in LLM training data

![how CodeAgent.run() operates, following the ReAct framework](https://cdn-lfs.hf.co/datasets/huggingface/documentation-images/6fd077476c038d7434c9450f1c8fd460b15d886eb083e13492547d7558d8be89?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27codeagent_docs.png%3B+filename%3D%22codeagent_docs.png%22%3B&response-content-type=image%2Fpng&Expires=1751666615&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MTY2NjYxNX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5oZi5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy82ZmQwNzc0NzZjMDM4ZDc0MzRjOTQ1MGYxYzhmZDQ2MGIxNWQ4ODZlYjA4M2UxMzQ5MjU0N2Q3NTU4ZDhiZTg5P3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=GA9K8mt5eUBMdVrCScQlj-%7E5Nzt9nE5Iv64lTnynXiaKcUsSqEkI-CQDrIjaHTZv8eD8QAaKQ1C-dNKc5LxkTXB5jvHFw0DdrC1Q8iDsVxL7kitQCP8wN-InQ8h5d8EIT7ENAR4Qb9p7yhHu1hDB1iDPuQXBBOIVnuq-EY5Xa4dfTKF65IzwpCTejtHw8GE%7E69YbyFq1OgwB9PBCjdgfCQBz2cbvBMB00gQ%7EGFZOzw%7EPOMrXW2O%7ELGgFm-Q3OOyl5%7E06F4lisMBc7cD5XWzJcNmC1It0Aq3tR8wZgbBnY13u6-Nof-w9DsA9zHCpkx0C1qMpqYbiGDri6XQz6p9xaA__&Key-Pair-Id=K3RPWS32NSSJCE)

A CodeAgent performs actions through a cycle of steps, with existing variables and knowledge being incorporated into the agent’s context, which is kept in an execution log:

The system prompt is stored in a SystemPromptStep, and the user query is logged in a TaskStep.

Then, the following while loop is executed:

2.1 Method agent.write_memory_to_messages() writes the agent’s logs into a list of LLM-readable chat messages.

2.2 These messages are sent to a Model, which generates a completion.

2.3 The completion is parsed to extract the action, which, in our case, should be a code snippet since we’re working with a CodeAgent.

2.4 The action is executed.

2.5 The results are logged into memory in an ActionStep.

At the end of each step, if the agent includes any function calls (in agent.step_callback), they are executed.

Code execution has strict security measures - imports outside a predefined safe list are blocked by default. However, you can authorize additional imports by passing them as strings in additional_authorized_imports. When creating the agent, we’ll use additional_authorized_imports to allow for importing the datetime module.

`smolagents` specializes in agents that write and execute Python code snippets, offering sandboxed execution for security. It supports both open-source and proprietary language models, making it adaptable to various development environments.


```
# Change to your username and repo name and push to hub
agent.push_to_hub('sergiopaniego/AlfredAgent')
# Change to your username and repo name and download from hub
alfred_agent = agent.from_hub('sergiopaniego/AlfredAgent', trust_remote_code=True)
# Run the agent
alfred_agent.run("Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme")
``` 


### Inspecting Our Party Preparator Agent with OpenTelemetry and Langfuse

Once again, smolagents comes to the rescue! It embraces the OpenTelemetry standard for instrumenting agent runs, allowing seamless inspection and logging. With the help of Langfuse and the SmolagentsInstrumentor, Alfred can easily track and analyze his agent’s behavior.

```
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents langfuse
```


```
import os

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..." 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..." 
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region
```

## Tool Calling Agent

