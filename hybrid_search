import os
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Any
import pandas as pd
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

class HybridSearchRAG:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        embedding_deployment: str,
        documents: List[Document] = None,
        content_key: str = "page_content",
    ):
        """
        Initialize the hybrid search system with langchain Document objects.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: Azure OpenAI API version
            embedding_deployment: Name of the embedding model deployment
            documents: List of langchain Document objects
            content_key: Key for the text content in Document objects (default: "page_content")
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.embedding_deployment = embedding_deployment
        self.documents = documents or []
        self.content_key = content_key
        
        # Create embeddings for all documents if documents are provided
        if documents:
            self.document_embeddings = self._get_embeddings(
                [getattr(doc, content_key) for doc in documents]
            )
            print(f"Initialized with {len(documents)} documents")
        else:
            self.document_embeddings = np.array([])
            print("Initialized without documents")
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to the search system.
        
        Args:
            documents: List of langchain Document objects to add
        """
        if not documents:
            return
        
        # Get content from documents
        contents = [getattr(doc, self.content_key) for doc in documents]
        
        # Get embeddings for new documents
        new_embeddings = self._get_embeddings(contents)
        
        # Add to existing documents and embeddings
        if self.documents:
            self.documents.extend(documents)
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
        else:
            self.documents = documents
            self.document_embeddings = new_embeddings
        
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts using Azure OpenAI."""
        embeddings = []
        
        # Process in batches to avoid API limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_deployment
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)
    
    def _keyword_filter(
        self, 
        documents: List[Document], 
        keywords: Union[str, List[str]]
    ) -> List[Document]:
        """
        Filter documents by keywords.
        
        Args:
            documents: List of Document objects to filter
            keywords: Single keyword or list of keywords to search for
        
        Returns:
            List of Document objects containing at least one of the keywords
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # Convert all keywords to lowercase for case-insensitive search
        keywords = [k.lower() for k in keywords]
        
        filtered_docs = []
        for doc in documents:
            text = getattr(doc, self.content_key).lower()
            if any(keyword in text for keyword in keywords):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _semantic_search(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            documents: List of Document objects to search through
            top_k: Number of top results to return
        
        Returns:
            List of tuples containing (document, similarity_score)
        """
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # If we're searching through a subset of all documents, get their indices
        if len(documents) < len(self.documents):
            indices = []
            doc_contents = [getattr(doc, self.content_key) for doc in documents]
            all_contents = [getattr(doc, self.content_key) for doc in self.documents]
            
            for content in doc_contents:
                if content in all_contents:
                    indices.append(all_contents.index(content))
            
            doc_embeddings = self.document_embeddings[indices]
        else:
            doc_embeddings = self.document_embeddings
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Get top-k results
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if len(documents) < len(self.documents):
                # Map back to the filtered documents
                doc_idx = min(idx, len(documents) - 1)
                doc = documents[doc_idx]
            else:
                doc = self.documents[idx]
            
            similarity = float(similarities[idx])
            results.append((doc, similarity))
        
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        keywords: Optional[Union[str, List[str]]] = None, 
        top_k: int = 5,
        semantic_only: bool = False
    ) -> Tuple[List[Tuple[Document, float]], bool]:
        """
        Perform hybrid search: keyword filtering followed by semantic search.
        
        Args:
            query: The search query for semantic search
            keywords: Optional keyword(s) for filtering (if None, only semantic search is used)
            top_k: Number of top results to return
            semantic_only: If True, bypasses keyword filtering and uses only semantic search
            
        Returns:
            Tuple of (search results, used_keywords_flag) where search results is a list of 
            tuples containing (document, similarity_score)
        """
        if not self.documents:
            print("No documents in the index. Please add documents first.")
            return [], False
        
        used_keywords = False
        
        # Check if keywords are provided and valid
        has_valid_keywords = (
            keywords is not None and 
            keywords not in ([], "", set(), {}, (), None) and
            not semantic_only
        )
        
        # If valid keywords are provided, filter documents first
        if has_valid_keywords:
            filtered_docs = self._keyword_filter(self.documents, keywords)
            if filtered_docs:
                used_keywords = True
                search_docs = filtered_docs
                print(f"Keyword filter applied. {len(filtered_docs)} documents matched.")
            else:
                # No documents matched the keywords, fall back to all documents
                search_docs = self.documents
                print("No documents matched keywords. Falling back to full semantic search.")
        else:
            # No keywords provided or semantic_only=True, use all documents
            search_docs = self.documents
            if semantic_only:
                print("Semantic search only mode. Keyword filtering bypassed.")
            else:
                print("No keywords provided. Using full semantic search.")
        
        # Apply semantic search
        results = self._semantic_search(query, search_docs, top_k=top_k)
        
        return results, used_keywords
    
    def keyword_only_search(
        self,
        keywords: Union[str, List[str]],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Perform keyword-only search without any semantic ranking.
        
        Args:
            keywords: Keyword(s) to search for
            top_k: Maximum number of results to return
            
        Returns:
            List of tuples containing (document, score) where score is always 1.0
        """
        if not self.documents:
            print("No documents in the index. Please add documents first.")
            return []
            
        if not keywords or keywords in ([], "", set(), {}, ()):
            print("Warning: No keywords provided for keyword search.")
            return []
            
        filtered_docs = self._keyword_filter(self.documents, keywords)
        if not filtered_docs:
            print("No documents matched the keywords.")
            return []
            
        # Return the filtered docs with a placeholder similarity of 1.0
        # Limited to top_k results
        return [(doc, 1.0) for doc in filtered_docs[:top_k]]
    
    def search(
        self,
        query: str = None,
        keywords: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> List[Tuple[Document, float]]:
        """
        Unified search interface that supports different search modes.
        
        Args:
            query: The search query (required for semantic and hybrid modes)
            keywords: Optional keyword(s) for filtering (required for keyword mode)
            top_k: Number of top results to return
            mode: Search mode - one of "hybrid", "semantic", or "keyword"
            
        Returns:
            List of tuples containing (document, similarity_score)
        """
        if not self.documents:
            print("No documents in the index. Please add documents first.")
            return []
        
        # Validate inputs based on search mode
        if mode.lower() in ["hybrid", "semantic"] and not query:
            print(f"Warning: {mode} mode requires a query but none provided.")
            return []
            
        if mode.lower() == "keyword" and not keywords:
            print("Warning: Keyword mode requires keywords but none provided.")
            return []
        
        # Execute search based on mode
        if mode.lower() == "hybrid":
            results, _ = self.hybrid_search(query, keywords, top_k)
            print(f"Performed hybrid search with query: '{query}' and keywords: {keywords}")
            return results
        
        elif mode.lower() == "semantic":
            results, _ = self.hybrid_search(query, keywords=None, top_k=top_k, semantic_only=True)
            print(f"Performed semantic-only search with query: '{query}'")
            return results
        
        elif mode.lower() == "keyword":
            results = self.keyword_only_search(keywords, top_k)
            print(f"Performed keyword-only search with keywords: {keywords}")
            return results
        
        else:
            raise ValueError(f"Invalid search mode: {mode}. Must be one of: hybrid, semantic, keyword")


# Example usage
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    # Set your Azure OpenAI credentials
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint")
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key")
    AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01")
    EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    # Create sample langchain documents
    sample_documents = [
        Document(
            page_content="Machine learning is a method of data analysis that automates analytical model building. It's a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.",
            metadata={"title": "Introduction to Machine Learning", "category": "AI", "id": "doc1"}
        ),
        Document(
            page_content="Deep learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Deep learning is behind many recent advances in AI, including computer vision and natural language processing.",
            metadata={"title": "Deep Learning Fundamentals", "category": "AI", "id": "doc2"}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages. NLP is used in many applications including chatbots, search engines, and sentiment analysis.",
            metadata={"title": "Natural Language Processing", "category": "AI", "id": "doc3"}
        ),
        Document(
            page_content="Cloud computing is the on-demand delivery of IT resources over the Internet with pay-as-you-go pricing. Instead of buying, owning, and maintaining physical data centers and servers, you can access technology services, such as computing power, storage, and databases, on an as-needed basis.",
            metadata={"title": "Cloud Computing Basics", "category": "Cloud", "id": "doc4"}
        ),
        Document(
            page_content="Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-Turbo, and Embeddings model series. These models can be easily adapted to your specific task including content generation, summarization, semantic search, and natural language to code translation.",
            metadata={"title": "Azure OpenAI Services", "category": "Cloud", "id": "doc5"}
        ),
        Document(
            page_content="Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It enhances large language models by retrieving relevant documents from a knowledge base before generating a response, allowing the model to access specific information that may not be in its training data.",
            metadata={"title": "Retrieval Augmented Generation", "category": "AI", "id": "doc6"}
        ),
        Document(
            page_content="Vector databases are specialized database systems designed to store and query vector embeddings efficiently. They're critical for semantic search applications, allowing for similarity search operations based on the meaning and context of the query rather than just keyword matching.",
            metadata={"title": "Vector Databases", "category": "Database", "id": "doc7"}
        ),
        Document(
            page_content="Hybrid search combines multiple search techniques to improve search relevance. By integrating keyword-based search with semantic search, it can leverage the strengths of both approaches: the precision of exact matching and the contextual understanding of embeddings-based similarity.",
            metadata={"title": "Hybrid Search Mechanisms", "category": "Search", "id": "doc8"}
        )
    ]
    
    def run_demo():
        """Run a demo of the HybridSearchRAG system with sample documents."""
        print("Initializing HybridSearchRAG with sample documents...")
        hybrid_search = HybridSearchRAG(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            embedding_deployment=EMBEDDING_DEPLOYMENT,
            documents=sample_documents
        )
        
        # Example 1: Using both keyword filtering and semantic search
        print("\n--- Example 1: Keyword filtering + semantic search (hybrid mode) ---")
        query = "How does Azure support AI models?"
        keywords = ["Azure", "OpenAI"]
        
        results = hybrid_search.search(
            query=query,
            keywords=keywords,
            mode="hybrid",
            top_k=3
        )
        
        print(f"Search query: '{query}'")
        print(f"Keywords: {keywords}")
        print("\nTop results:")
        
        for i, (doc, similarity) in enumerate(results):
            print(f"\n--- Result {i+1} (Similarity: {similarity:.4f}) ---")
            print(f"Title: {doc.metadata.get('title', 'No title')}")
            print(f"Category: {doc.metadata.get('category', 'No category')}")
            print(f"Content: {doc.page_content[:150]}...")
        
        # Example 2: Semantic-only search
        print("\n\n--- Example 2: Semantic-only search ---")
        query = "What is machine learning?"
        
        results = hybrid_search.search(
            query=query,
            mode="semantic",
            top_k=3
        )
        
        print(f"Search query: '{query}'")
        print("\nTop results:")
        
        for i, (doc, similarity) in enumerate(results):
            print(f"\n--- Result {i+1} (Similarity: {similarity:.4f}) ---")
            print(f"Title: {doc.metadata.get('title', 'No title')}")
            print(f"Category: {doc.metadata.get('category', 'No category')}")
            print(f"Content: {doc.page_content[:150]}...")
        
        # Example 3: Keyword-only search
        print("\n\n--- Example 3: Keyword-only search ---")
        keywords = ["Azure", "cloud"]
        
        results = hybrid_search.search(
            keywords=keywords,
            mode="keyword",
            top_k=3
        )
        
        print(f"Keywords: {keywords}")
        print("\nTop results:")
        
        for i, (doc, similarity) in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Title: {doc.metadata.get('title', 'No title')}")
            print(f"Category: {doc.metadata.get('category', 'No category')}")
            print(f"Content: {doc.page_content[:150]}...")
    
    # Run the demo if environment variables are set
    if AZURE_OPENAI_ENDPOINT != "your-azure-openai-endpoint" and AZURE_OPENAI_API_KEY != "your-azure-openai-api-key":
        run_demo()
    else:
        print("Set your Azure OpenAI credentials in environment variables to run the demo.")
        print("Required environment variables:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_API_VERSION (optional, defaults to 2023-12-01)")
        print("  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT (optional, defaults to text-embedding-ada-002)")
