"""
LLM Agent with RAG Tool Integration

This module defines the customer support agent that uses a Language Model
with Retrieval-Augmented Generation (RAG) capabilities.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
from typing import Optional, Dict, Any, List, AsyncIterator, cast

import httpx
import dotenv

dotenv.load_dotenv()




class BaseAgent(ABC):
    """Abstract base class for LLM agents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        self.memory: List[Dict[str, str]] = []

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def process_query(self, text: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass


class CustomerSupportAgent(BaseAgent):
    """
    Customer Support Agent with direct LLM + RAG flow.

    Uses ChromaDB as the vector store and Groq as the LLM provider.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm = None
        self.knowledge_base = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.logger = logging.getLogger(__name__)
        self.provider = str(
            self.config.get("provider") or os.getenv("LLM_PROVIDER", "groq")
        ).lower()
        configured_model = self.config.get("model") or os.getenv("LLM_MODEL")
        provider_default_model = self._default_model_for_provider(self.provider)
        if configured_model:
            self.model = configured_model
        elif self.provider == "groq":
            self.model = os.getenv("GROQ_LLM_MODEL") or provider_default_model
        else:
            self.model = provider_default_model
        self.temperature = float(
            self.config.get("temperature", os.getenv("LLM_TEMPERATURE", 0.2))
        )
        self.timeout_seconds = float(
            self.config.get("timeout_seconds", os.getenv("LLM_TIMEOUT_SECONDS", 30.0))
        )

    async def initialize(self) -> None:
        """Initialize the LLM HTTP client and the optional ChromaDB knowledge base."""
        if self.provider == "groq":
            api_key = (
                self.config.get("api_key")
                or os.getenv("LLM_API_KEY")
                or os.getenv("GROQ_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "LLM API key not provided. Set config['api_key'] or "
                    "LLM_API_KEY (fallback: GROQ_API_KEY)."
                )
            base_url = "https://api.groq.com/openai/v1"
        elif self.provider == "openai":
            api_key = (
                self.config.get("api_key")
                or os.getenv("LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "LLM API key not provided. Set config['api_key'] or "
                    "LLM_API_KEY (fallback: OPENAI_API_KEY)."
                )
            base_url = "https://api.openai.com/v1"
        else:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. Supported providers: groq, openai."
            )

        self.http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(self.timeout_seconds),
        )

        await self._setup_knowledge_base()
        self.is_initialized = True

    async def _setup_knowledge_base(self) -> None:
        """
        Set up the ChromaDB knowledge base with embedded customer support documents.

        If a persistent collection already exists and is populated, it is reused
        to avoid re-embedding on every startup.
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            db_path = "./data/chroma_db"
            os.makedirs(db_path, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(path=db_path)
            collection_name = "customer_support_kb"

            # Reuse existing collection if it already has documents
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                if self.collection.count() > 0:
                    self.logger.info(
                        "Knowledge base already exists with %s documents",
                        self.collection.count(),
                    )
                    return
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Customer support knowledge base"},
                )

            knowledge_documents = self._get_customer_support_documents()
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            self.logger.info(
                "Ingesting %s documents into the knowledge base...",
                len(knowledge_documents),
            )

            import hashlib

            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []

            for i, doc_data in enumerate(knowledge_documents):
                doc_id = (
                    f"doc_{i}_{hashlib.md5(doc_data['content'].encode()).hexdigest()[:8]}"
                )
                documents.append(doc_data["content"])
                metadatas.append(
                    {
                        "category": doc_data["category"],
                        "title": doc_data["title"],
                        "doc_id": doc_id,
                    }
                )
                ids.append(doc_id)

            self.collection.add(
                documents=documents,
                metadatas=cast(Any, metadatas),
                ids=ids,
            )
            self.logger.info(
                "Successfully ingested %s documents into ChromaDB",
                len(documents),
            )

        except Exception as e:
            self.chroma_client = None
            self.collection = None
            self.embedding_model = None
            self.logger.warning(
                "Knowledge base unavailable. Continuing without RAG context: %s",
                e,
            )

    def _get_customer_support_documents(self) -> List[Dict[str, str]]:
        """Predefined customer support knowledge base (16 documents)."""
        return [
            # ── Returns ──────────────────────────────────────────────────────
            {
                "title": "Return Policy Overview",
                "category": "returns",
                "content": (
                    "We offer a 30-day return policy for all products purchased from our "
                    "store. Items must be in original condition with all tags and packaging "
                    "intact. Returns are processed within 5-7 business days of receiving "
                    "the returned item. Refunds are issued to the original payment method."
                ),
            },
            {
                "title": "Return Process Steps",
                "category": "returns",
                "content": (
                    "To initiate a return: 1) Log into your account and go to Order History, "
                    "2) Select the order and click 'Return Items', 3) Choose the items to "
                    "return and reason, 4) Print the prepaid return label, 5) Pack items "
                    "securely and attach the label, 6) Drop off at any UPS location or "
                    "schedule pickup."
                ),
            },
            {
                "title": "Non-Returnable Items",
                "category": "returns",
                "content": (
                    "The following items cannot be returned: personalized or customized "
                    "products, perishable goods, digital downloads, gift cards, intimate "
                    "apparel, and items marked as final sale. Health and safety regulations "
                    "prevent returns of opened cosmetics and personal care items."
                ),
            },
            # ── Shipping ─────────────────────────────────────────────────────
            {
                "title": "Shipping Methods and Times",
                "category": "shipping",
                "content": (
                    "We offer multiple shipping options: Standard shipping (5-7 business "
                    "days, free on orders over $50), Express shipping (2-3 business days, "
                    "$12.99), Next-day shipping (1 business day, $24.99). All orders placed "
                    "before 2 PM EST ship the same day."
                ),
            },
            {
                "title": "International Shipping",
                "category": "shipping",
                "content": (
                    "We ship internationally to over 50 countries. International shipping "
                    "takes 7-14 business days via DHL Express. Shipping costs vary by "
                    "destination and are calculated at checkout. Customers are responsible "
                    "for customs fees and import duties. Some restrictions apply to certain "
                    "products and countries."
                ),
            },
            {
                "title": "Order Tracking",
                "category": "shipping",
                "content": (
                    "Once your order ships, you'll receive a tracking number via email. "
                    "Track your package using the tracking number on our website or the "
                    "carrier's website. You can also track orders by logging into your "
                    "account and viewing Order History. Tracking updates may take 24 hours "
                    "to appear."
                ),
            },
            # ── Support ──────────────────────────────────────────────────────
            {
                "title": "Contact Information",
                "category": "support",
                "content": (
                    "Customer support is available 24/7 via multiple channels: "
                    "Phone: 1-800-HELP-NOW (1-800-435-7669), "
                    "Email: support@company.com, "
                    "Live chat on our website (available 6 AM - 12 AM EST), or submit a "
                    "support ticket through your account dashboard."
                ),
            },
            {
                "title": "Response Times",
                "category": "support",
                "content": (
                    "Our support team response times: Live chat - immediate during business "
                    "hours, Phone support - average wait time under 3 minutes, "
                    "Email support - response within 4 hours during business days, "
                    "Support tickets - response within 24 hours. Premium customers receive "
                    "priority support with faster response times."
                ),
            },
            # ── Warranty / Technical ─────────────────────────────────────────
            {
                "title": "Product Warranty",
                "category": "warranty",
                "content": (
                    "All products come with a manufacturer's warranty. Electronics have "
                    "1-year warranty covering defects and malfunctions. Apparel and "
                    "accessories have 90-day warranty against material defects. Warranty "
                    "claims require proof of purchase and must be initiated within the "
                    "warranty period."
                ),
            },
            {
                "title": "Technical Support",
                "category": "technical",
                "content": (
                    "Free technical support is available for all electronic products. Our "
                    "certified technicians provide assistance with setup, troubleshooting, "
                    "and software issues. Technical support is available Monday-Friday "
                    "8 AM - 8 PM EST via phone or email. We also offer remote assistance "
                    "for compatible devices."
                ),
            },
            # ── Account / Orders ─────────────────────────────────────────────
            {
                "title": "Account Management",
                "category": "account",
                "content": (
                    "Manage your account online: Update personal information and addresses, "
                    "view order history and tracking, manage payment methods, set "
                    "communication preferences, download invoices and receipts. Account "
                    "changes may take up to 24 hours to reflect across all systems."
                ),
            },
            {
                "title": "Order Modifications",
                "category": "orders",
                "content": (
                    "Orders can be modified or canceled within 1 hour of placement if not "
                    "yet processed. Contact customer service immediately to make changes. "
                    "Once an order is processed and shipped, it cannot be modified. You can "
                    "return unwanted items following our return policy."
                ),
            },
            # ── Payment / Billing ────────────────────────────────────────────
            {
                "title": "Payment Methods",
                "category": "payment",
                "content": (
                    "We accept all major credit cards (Visa, MasterCard, American Express, "
                    "Discover), PayPal, Apple Pay, Google Pay, and Buy Now Pay Later "
                    "options through Klarna and Afterpay. Gift cards and store credit can "
                    "also be used for purchases. Payment is processed securely using "
                    "256-bit SSL encryption."
                ),
            },
            {
                "title": "Billing and Invoices",
                "category": "billing",
                "content": (
                    "Billing occurs when your order ships. You'll receive an email "
                    "confirmation with invoice details. Invoices are available in your "
                    "account under Order History. For business purchases, we can provide "
                    "detailed invoices with tax information. Contact our billing department "
                    "for any payment disputes or questions."
                ),
            },
            # ── Products ─────────────────────────────────────────────────────
            {
                "title": "Product Availability",
                "category": "products",
                "content": (
                    "Product availability is updated in real-time on our website. If an "
                    "item shows as 'In Stock', it's available for immediate shipping. "
                    "'Limited Stock' means fewer than 10 items remaining. 'Pre-order' "
                    "items will ship on the specified release date. Out of stock items can "
                    "be added to your wishlist for restock notifications."
                ),
            },
            {
                "title": "Size and Fit Guide",
                "category": "products",
                "content": (
                    "Each product page includes detailed size charts and fit information. "
                    "For apparel, we recommend checking measurements against our size guide "
                    "rather than relying on size labels from other brands. If you're "
                    "between sizes, we generally recommend sizing up. Our customer service "
                    "team can provide personalized fit recommendations."
                ),
            },
        ]

    async def _rag_search(self, query: str) -> str:
        """
        Retrieve the top-3 most relevant documents from ChromaDB for a given query.

        Returns a formatted string with titles, categories, relevance scores, and
        document content, ready to be injected into the LLM system prompt.
        """
        if not hasattr(self, "collection") or self.collection is None:
            return (
                "Knowledge base not available. "
                "Please ensure the service is properly initialized."
            )

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            docs_result = results.get("documents") or [[]]
            metas_result = results.get("metadatas") or [[]]
            distances_result = results.get("distances") or [[]]

            documents = docs_result[0] if docs_result else []
            metadatas = metas_result[0] if metas_result else []
            distances = distances_result[0] if distances_result else []

            if not documents:
                return "I couldn't find relevant knowledge-base information for this query."

            formatted: List[str] = []
            for doc, meta, distance in zip(documents, metadatas, distances):
                meta_map = cast(Dict[str, Any], meta or {})
                title = meta_map.get("title", "Untitled")
                category = meta_map.get("category", "general")
                # Lower distance = higher relevance
                formatted.append(
                    f"[{title}] ({category}, relevance={1 - distance:.3f}):\n{doc}"
                )

            return "\n\n".join(formatted)

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    async def process_query(self, text: str, **kwargs) -> str:
        """
        Process a user query via RAG retrieval + Groq LLM inference.

        Returns the complete agent response as a string.
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")

        chunks: List[str] = []
        async for token in self.stream_query(text, **kwargs):
            chunks.append(token)
        return "".join(chunks).strip()

    async def stream_query(self, text: str, **kwargs) -> AsyncIterator[str]:
        """
        Stream model output tokens. Retrieves RAG context first, then calls Groq.
        """
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized")
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")

        rag_context = await self._rag_search(text)

        system_prompt = (
            "You are a helpful and friendly customer support assistant. "
            "Use the knowledge context below to answer accurately. "
            "If the context doesn't cover the question, say so politely and "
            "offer to connect the customer with a human agent.\n\n"
            f"Knowledge context:\n{rag_context}"
        )

        async for token in self._groq_chat_stream(
            user_message=text, system_prompt=system_prompt, **kwargs
        ):
            yield token

    async def cleanup(self) -> None:
        """Close the Groq HTTP client and reset state."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        self.llm = None
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.is_initialized = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _groq_chat_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Call the Groq chat completions endpoint in streaming mode."""
        if not self.http_client:
            raise RuntimeError("Groq HTTP client not initialized")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        body = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        async with self.http_client.stream(
            "POST", "/chat/completions", json=body
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    @staticmethod
    def _default_model_for_provider(provider: str) -> str:
        if provider == "openai":
            return "gpt-4o-mini"
        return "llama-3.1-8b-instant"
