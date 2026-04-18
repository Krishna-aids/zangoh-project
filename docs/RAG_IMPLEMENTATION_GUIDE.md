# RAG Implementation Guide

This guide explains how to implement the Retrieval-Augmented Generation (RAG) functionality for the customer support agent.

## What's Already Done for You

### Knowledge Base Setup (Complete)
- **16 comprehensive customer support documents** covering:
  - Return policies and procedures
  - Shipping information and tracking
  - Customer support contact methods
  - Warranty and technical support
  - Account management
  - Payment and billing
  - Product information

### Data Ingestion Pipeline (Complete)
- Automatic document ingestion into ChromaDB
- Embeddings generated using `sentence-transformers`
- Persistent storage (data survives server restarts)
- Duplicate prevention (won't re-ingest existing data)

### Infrastructure (Complete)
- ChromaDB setup with persistent client
- SentenceTransformer model (`all-MiniLM-L6-v2`)
- Document metadata management (titles, categories)
- Error handling and logging

## Your Task: Implement RAG Search

You need to complete **ONE function**: `_rag_search()` in `src/llm/agent.py`

### Function Signature
```python
async def _rag_search(self, query: str) -> str:
```

**Input**: User query (e.g., "What is your return policy?")  
**Output**: Formatted response with relevant information

## Implementation Steps

### Step 1: Query ChromaDB
```python
# Use ChromaDB's built-in query method
results = self.collection.query(
    query_texts=[query],        # The user's question
    n_results=3,                # Number of relevant documents to retrieve
    include=['documents', 'metadatas', 'distances']
)
```

### Step 2: Extract Results
```python
# Results structure:
documents = results['documents'][0]      # List of document contents
metadatas = results['metadatas'][0]      # List of metadata (title, category)  
distances = results['distances'][0]      # Similarity scores (lower = more similar)
```

### Step 3: Format Response
```python
# Example formatting:
formatted_results = []
for doc, meta, distance in zip(documents, metadatas, distances):
    formatted_results.append(f"**{meta['title']}** ({meta['category']})\n{doc}")

return "\n\n".join(formatted_results)
```

## Complete Implementation Example

```python
async def _rag_search(self, query: str) -> str:
    if not hasattr(self, 'collection') or self.collection is None:
        return "Knowledge base not available."
    
    try:
        # Search for relevant documents
        results = self.collection.query(
            query_texts=[query],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Check if we found any results
        if not results['documents'] or not results['documents'][0]:
            return "I couldn't find relevant information for your query."
        
        # Format the results
        formatted_results = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append(
                f"**{meta['title']}** (Category: {meta['category']})\n{doc}"
            )
        
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
```

## Testing Your Implementation

### Method 1: Use the Test Script
```bash
cd audio_support_agent
python src/utils/kb_test.py
```

This script will:
- Set up the knowledge base
- Show sample queries
- Test your `_rag_search()` implementation
- Display knowledge base structure

### Method 2: Manual Testing
```python
# In your test file or Python REPL
from src.llm.agent import CustomerSupportAgent

agent = CustomerSupportAgent({})
await agent._setup_knowledge_base()

# Test your implementation
result = await agent._rag_search("What is your return policy?")
print(result)
```

### Method 3: API Testing
Once you complete the pipeline, test via the API:
```bash
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?"}'
```

## Sample Queries for Testing

Use these queries to test your implementation:

1. **Return Policy**: "What is your return policy?"
2. **Shipping**: "How long does shipping take?"
3. **Contact**: "How can I contact customer support?"
4. **Payment**: "What payment methods do you accept?"
5. **International**: "Do you ship internationally?"
6. **Warranty**: "What warranty do you offer?"
7. **Tracking**: "How do I track my order?"
8. **Cancellation**: "Can I cancel my order?"
9. **Technical Support**: "Do you offer technical support?"
10. **Account**: "How do I manage my account?"

## Advanced Implementation Ideas

### Basic Implementation (Required)
- Simple query and format results
- Return top 3 most relevant documents

### Enhanced Implementation (Bonus)
```python
# Filter by similarity threshold
relevant_docs = [
    (doc, meta) for doc, meta, distance in zip(documents, metadatas, distances)
    if distance < 0.8  # Only include highly relevant results
]

# Category-based formatting
results_by_category = {}
for doc, meta in relevant_docs:
    category = meta['category']
    if category not in results_by_category:
        results_by_category[category] = []
    results_by_category[category].append(f"**{meta['title']}**: {doc}")

# Format by category
formatted_response = []
for category, docs in results_by_category.items():
    formatted_response.append(f"## {category.title()}\n" + "\n".join(docs))

return "\n\n".join(formatted_response)
```

### Expert Implementation (Bonus)
- Query expansion for better matching
- Hybrid search (keyword + semantic)
- Response summarization using LLM
- Context-aware filtering

## Common Issues and Solutions

### Issue: "Collection not found"
**Solution**: Ensure `_setup_knowledge_base()` is called during agent initialization.

### Issue: Empty results
**Solution**: Check if ChromaDB has data using `self.collection.count()`.

### Issue: Poor search quality
**Solutions**:
- Adjust `n_results` parameter
- Add similarity threshold filtering
- Try query preprocessing (lowercasing, removing stopwords)

### Issue: Formatting looks bad
**Solutions**:
- Add markdown formatting (`**bold**`, `## headers`)
- Use proper line breaks (`\n\n`)
- Include document titles and categories

## Evaluation Criteria

Your RAG implementation will be evaluated on:

1. **Functionality** (60%): Does it retrieve relevant documents?
2. **Code Quality** (25%): Clean, readable implementation
3. **Error Handling** (10%): Proper exception handling
4. **Formatting** (5%): Readable response formatting

## Testing Commands Summary

```bash
# Test knowledge base setup
python src/utils/kb_test.py

# Test full pipeline (after implementing other components)
python -m src.api.server

# Test via API
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?"}'
```

## Tips for Success

1. **Start Simple**: Get basic retrieval working first
2. **Test Incrementally**: Test each step of your implementation
3. **Use the Test Script**: It shows you exactly what's in the knowledge base
4. **Check ChromaDB Docs**: [ChromaDB Query Documentation](https://docs.trychroma.com/usage-guide#querying-a-collection)
5. **Handle Edge Cases**: Empty queries, no results, errors
6. **Format Nicely**: Users need to read your responses!

Good luck implementing your RAG system!