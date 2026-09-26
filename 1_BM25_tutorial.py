#%%
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import bm25s

# from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample documents
docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

print(f"loaded {len(docs)} documents")

# 1. Tokenize corpus text
corpus_text = [doc.page_content for doc in docs]
corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en")

print(corpus_tokens.ids[:1])  # list[list[int]] -- one inner list per doc
print(list(corpus_tokens.vocab.items())[:10])  # dict[str, int] -- token string -> integer ID

# 2. Initialize and build the BM25S index
retriever = bm25s.BM25(corpus=docs)
retriever.index(corpus_tokens)

# 3. Query the index
# query = "What frameworks exist for building LLM applications?"
query="What databases are used for AI applications?"
query_tokens = bm25s.tokenize([query], stopwords='en')

# 4. Retrieve top-k documents and scores
results, scores = retriever.retrieve(query_tokens, k=4)

print(f"Query: '{query}'\n")
for idx, (doc, score) in enumerate(zip(results[0], scores[0]), start=1):
    print(f"(Score: {score:.4f}):")
    print(f"Content:  {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")

    


