
Before starting up with [src/graphs/corrective_rag_example.py](corrective_rag_example.py)




Add the following config file configs/openai-rag-corrective.json with the next content

```json
{
    "api_token": "<your token api>",
    "tavily_api_key": "<your token api>",
    "vectordb_path": "<location for chroma db>",
    "collection_name": "<name for db you would like to use"
}
```

Run ingestion script to fill vector databse with small amount of data for demo:

```console
python src/graphs/corrective_rag/corrective_rag_example_ingestion.py
```

See docstrings in [src/graphs/corrective_rag_example.py](corrective_rag_example.py) for the next steps!