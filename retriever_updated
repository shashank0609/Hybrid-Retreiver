from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
import openai
from openai import AzureOpenAI
import numpy as np

class HybridRetriever:
    def __init__(
        self,
        documents: List[str],
        azure_endpoint: str,
        azure_api_key: str,
        azure_deployment: str,
        api_version: str = "2024-02-15-preview",
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ):
        """
        Initialize the hybrid retriever with BM25 and Azure OpenAI embeddings.
        
        Args:
            documents: List of text documents to index
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name for embeddings
            api_version: Azure OpenAI API version
            bm25_weight: Weight for BM25 scores (default: 0.5)
            semantic_weight: Weight for semantic search scores (default: 0.5)
        """
        if not (0 <= bm25_weight <= 1 and 0 <= semantic_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if abs(bm25_weight + semantic_weight - 1) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )
        self.azure_deployment = azure_deployment
        
        # Initialize BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Generate and store document embeddings
        self.document_embeddings = self._get_embeddings(documents)
        
        # Store original documents
        self.documents = documents
        
        # Initialize score scalers
        self.bm25_scaler = MinMaxScaler()
        self.semantic_scaler = MinMaxScaler()
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using Azure OpenAI.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        embeddings = []
        # Process in batches of 16 to avoid rate limits
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.client.embeddings.create(
                model=self.azure_deployment,
                input=batch
            )
            embeddings.extend([e.embedding for e in batch_embeddings.data])
        
        return np.array(embeddings)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between two sets of vectors.
        
        Args:
            a: First set of vectors
            b: Second set of vectors
            
        Returns:
            numpy.ndarray: Array of cosine similarities
        """
        return np.dot(a, b.T) / (
            np.linalg.norm(a, axis=1).reshape(-1, 1) *
            np.linalg.norm(b, axis=1)
        )
        
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Any]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            return_scores: Whether to return scores along with documents
            
        Returns:
            If return_scores is False: List of retrieved documents
            If return_scores is True: List of tuples (document, score)
        """
        # Get BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_scores = self.bm25_scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        
        # Get semantic scores
        query_embedding = self._get_embeddings([query])[0]
        semantic_scores = self._cosine_similarity(
            self.document_embeddings,
            query_embedding.reshape(1, -1)
        ).flatten()
        semantic_scores = self.semantic_scaler.fit_transform(
            semantic_scores.reshape(-1, 1)
        ).flatten()
        
        # Combine scores
        combined_scores = (
            self.bm25_weight * bm25_scores +
            self.semantic_weight * semantic_scores
        )
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        if return_scores:
            return [
                (self.documents[i], combined_scores[i])
                for i in top_indices
            ]
        return [self.documents[i] for i in top_indices]

    def retrieve_with_details(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with detailed scoring information.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing document, combined score,
            and individual scores
        """
        # Get BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_scores = self.bm25_scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        
        # Get semantic scores
        query_embedding = self._get_embeddings([query])[0]
        semantic_scores = self._cosine_similarity(
            self.document_embeddings,
            query_embedding.reshape(1, -1)
        ).flatten()
        semantic_scores = self.semantic_scaler.fit_transform(
            semantic_scores.reshape(-1, 1)
        ).flatten()
        
        # Combine scores
        combined_scores = (
            self.bm25_weight * bm25_scores +
            self.semantic_weight * semantic_scores
        )
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        return [
            {
                "document": self.documents[i],
                "combined_score": combined_scores[i],
                "bm25_score": bm25_scores[i],
                "semantic_score": semantic_scores[i]
            }
            for i in top_indices
        ]

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps across a sleeping canine",
        "The cat and dog are playing together",
        "Machine learning models process data efficiently",
        "Natural language processing helps computers understand text"
    ]
    
    # Initialize retriever with Azure OpenAI credentials
    retriever = HybridRetriever(
        documents=documents,
        azure_endpoint="YOUR_AZURE_ENDPOINT",
        azure_api_key="YOUR_API_KEY",
        azure_deployment="YOUR_DEPLOYMENT_NAME",
        bm25_weight=0.3,
        semantic_weight=0.7
    )
    
    # Simple retrieval
    query = "fox jumping over dog"
    results = retriever.retrieve(query, top_k=2)
    print("\nTop 2 results for query:", query)
    for doc in results:
        print("-", doc)
    
    # Detailed retrieval with scores
    detailed_results = retriever.retrieve_with_details(query, top_k=2)
    print("\nDetailed results:")
    for result in detailed_results:
        print("\nDocument:", result["document"])
        print("Combined score:", round(result["combined_score"], 3))
        print("BM25 score:", round(result["bm25_score"], 3))
        print("Semantic score:", round(result["semantic_score"], 3))
