# Skeernir

... to be explained soon...

The project in an intensive development phase. 
Here's what you need to do to get your environment set up and get started:

## Setup Conda Environment with Python 3.11

1. Create a new conda environment with Python 3.11:
```console
conda create --name my_env python=3.11 
conda activate my_env
```
2. Update packages and install `ccache` and `cmake`:

```console
apt-get update
apt-get install ccache
apt-get install cmake
```

## Prepare for Graphs

Install Llama.cpp to use llama.cpp server .See [Llama.cpp](https://github.com/ggerganov/llama.cpp) for detailed instructions.
The same to [Ollama](https://github.com/ggerganov/llama.cpp) or [vLLM](https://github.com/vllm-project/vllm) 

You can communicate with your models and tools as you prefer, just implement your Langgraph Graph based on your previous projects or based on examples provided. 
For instance if you're going to work with LLM models using llama.cpp python bindings, perform the following steps:

1. In your newly created conda environment, install llama-cpp-python:

```console
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```
If GPU on board:

```console
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```
2. Install all dependencies from the requirements.txt file:
```console
pip install -r requirements.txt --no-cache-dir
```

## Add / configure your Graphs

Beckend supports graphs defined in Langgraph, see examples in `/src/graphs`. 
Decorate function using `@tool_graph` which returns compiled graph and restart app - you Agentic solution will be available in UI. Find more details in tutorial (here would be a link)
Configure your models for use in both vendor-provided LLMs and locally-deployed instances:

1. Vendor LLMs (e.g., GPT-4o-mini or Sonnet3.5)
For LLMs provided by vendors like GPT-4o-mini, add the following configs - .configs/gpt-4o-mini-default.json and .configs/Sonnet35.json
```json
{
    "api_token": "<your token api>"
}
```
Replace <your_token_api> with your actual API token.

For example with corrective_rag add the following config - .configs/openai-rag-corrective.json
```json
{
    "api_token": "<your token api>",
    "tavily_api_key": "<your token api>",
    "vectordb_path": "<location for chroma db>",
    "collection_name": "<name for db you would like to use"
}
```

Then run ingestion script to fill vector databse with small amount of data for demo:

```console
python src/graphs/corrective_rag/corrective_rag_example_ingestion.py
```


2. Locally-deployed Models
For locally deployed models, refer to the following Python scripts as examples:

* .src/graphs/llama3_2_vision_11b_on_ollama_server.py
* .src/graphs/phi3_5_mini_instruct_on_llamacpp_server.py

and corresponding .json files in .configs/

3. Download GGUF Files (or other formats in case you are using vLLM) for your models and edit configs accordingly. 
Models for Ollama will be pulled automatically if not available. 


## HTTPS

Create certificates with openssl

https://medium.com/@mariovanrooij/adding-https-to-fastapi-ad5e0f9e084e

Then change path to key.pem and cert.pem in main.py

```python

    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8899,
            loop="asyncio",
            log_level="info",
            access_log=True,
            ssl_keyfile="<path to key.pem>",
            ssl_certfile="<path to cert.pem>",
            ssl_keyfile_password="<your pass>")
    
    except Exception as e:
        logger.info(f"An error occurred: {e}")
```

Run 

```concole 
python main.py
```



