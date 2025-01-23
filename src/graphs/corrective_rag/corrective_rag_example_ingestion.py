import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings


if __name__ == "__main__":

    import sys
    import json

    # for testing purposes:
    p = "/".join(sys.path[0].split('/')[:-3])
    if p not in sys.path:
        sys.path.append(p)

    with open(os.path.join(p, 'configs/openai-rag-corrective.json'), 'r') as f:
        data = json.load(f)

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=data["collection_name"],
        embedding=OpenAIEmbeddings(api_key=data['api_token']),
        persist_directory=data["vectordb_path"],
    )