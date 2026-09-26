#%%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

documents = [
    Document (
        page_content = 'Product SKU-774X is your flagship router.'
        'It supports gigabit speeds and advanced QoS features.',
        metadata={'type':'product'}
    ),
    Document (
        page_content = 'For network connectivity issues, first check the '
        'ethernet cable and router status lights.',
        metadata={'type':'troubleshooting'}
    ),
    Document (
        page_content = 'Error code E_CONN_REFUSED indicates the server'
        'Rejected the connection. check firewall settings ',
        metadata={'type':'error'}
    ),
    Document (
        page_content = 'Router configuration guide: Access the admin panel '
        'at 192.168.1.1 to modify settings.',
        metadata={'type':'config'}
    ),
    Document (
        page_content = 'The authentication process requires valid credentials.'
        'Use OAuth2 for secure API access. ',
        metadata={'type':'auth'}
    ),
    Document (
        page_content = 'WCAG 2.1 compliance requires all images to have '
        'alt text and sufficient color contrast.',
        metadata={'type':'compliance'}
    )
]

print(f"loaded {len(documents)} documents")

#Create vector store
vectorstore = Chroma.from_documents(
    documents, embeddings, collection_name='hybrid_test'
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs={'k':3}
)

print("Vector retriever ready!!")

bm25_retriever = BM25Retriever.from_documents(
    documents, k=3
)

print("BM25 retriever ready!!")

#Combine with EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights = [0.5, 0.5]
)

print("Hybrid retriever is ready!!")

def test_query(query, name, retriever):
    '''Test a query and show results'''
    results = retriever.invoke(query)
    print(f"\\n {name} => Query: \"{query}\" ")

    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'{i+1}, {preview}')
    return results

queries = [
    'SKU-774X specifications ',
    'E_CONN_REFUSED error',
    'How do I authenticate?',
    'WCAG compliance',
    'router configuration'
]

for query in queries:
    print("="*60)

    # Vector only
    vector_results = test_query(query, 'VECTOR', vector_retriever)

    # BM25 only
    bm25_results = test_query(query, 'BM25', bm25_retriever)

    # Hybrid results 
    hybrid_results = test_query(query, 'HYBRID', ensemble_retriever)


def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    '''Combine multiple retrievers using weighted reciprocal rank fusion'''
    doc_scores = {} #page_content => (rrf_score, doc)

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1 / (rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key= lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]] 

from typing import List
class HybridRetriever:
    '''Production hybrid retriever with BM25 + vector search'''
    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int = 4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        self.vectorstore = Chroma.from_documents(
                documents, self.embeddings, collection_name='hybrid_test'
            )
        self.vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={'k': k}
            )

        self.bm25_retriever = BM25Retriever.from_documents(
                documents, k=k
            )

    def search(self, query: str) -> List[Document]:
        '''Run hybrid search using weighted RRF'''
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights = [self.bm25_weight, self.vector_weight],
            k = self.k
        )

    def add_documents(self, documents: List[Document]):
        '''Add new documents to both retrievers'''
        self.vectorstore.add_documents(documents)

        all_docs = self.vectorstore.get()
        self.bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=content, metadata=meta or {}) for content, meta in zip(all_docs["documents"], all_docs["metadatas"])], k=self.k
        )        
    
retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
results = retriever.search("SKU-774X specifications")

for doc in results:
    print(doc.page_content[:100])
        
# %%
