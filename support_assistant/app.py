import os
from pathlib import Path
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, StateGraph

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"

# Mock mode is the required offline mode and is the default.
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

# Exact policy corpus from the project brief.
POLICY_DOCS = {
    "doc_01": (
        "Zepto delivers grocery and household essentials to serviceable pin codes "
        "within 10 to 30 minutes of order confirmation, depending on the customer's "
        "delivery zone and current order volume. Standard delivery is free on orders "
        "over INR 149; orders below this threshold incur a flat INR 25 delivery fee. "
        "Priority delivery, which reserves the next available rider slot, is available "
        "at checkout for an additional INR 15. Zepto does not currently deliver to "
        "addresses outside its listed serviceable pin codes."
    ),
    "doc_02": (
        "Grocery and perishable items may be reported for a return within 24 hours "
        "of delivery if damaged, spoiled, or incorrect; non-perishable packaged items "
        "may be returned within 7 days of delivery in unopened, resalable condition. "
        "Approved refunds are credited to the original payment method within 3–5 "
        "business days, or instantly to the Zepto wallet if the customer opts for "
        "wallet credit. Personal care items that have been opened are non-returnable "
        "except in the case of a manufacturing defect. Return pickup, where required, "
        "is arranged free of cost by Zepto."
    ),
    "doc_03": (
        "Zepto offers three account tiers: Basic (free, default tier, standard delivery "
        "fees apply), Zepto Pass (INR 49 per month, free standard delivery on all "
        "orders and 5% off select categories), and Zepto Pass+ (INR 99 per month, "
        "free priority delivery, 10% off select categories, and early access to "
        "limited-time deals 24 hours before they go live to Basic and Pass members). "
        "Membership can be cancelled at any time from account settings; cancelling "
        "stops the next billing cycle but does not refund the current membership period."
    ),
    "doc_04": (
        "Every Zepto order shows a live rider-tracking map from the moment it is packed "
        "until delivery, accessible from the 'Track Order' screen. Estimated delivery "
        "time updates automatically as the rider moves. If an order's status shows no "
        "movement for more than 20 minutes past its original estimated delivery time, "
        "customers should contact support directly rather than continue waiting, since "
        "this indicates a likely delivery issue."
    ),
    "doc_05": (
        "Orders can be cancelled free of cost any time before the order status changes "
        "to 'Packed', typically within the first 2 minutes of placing the order. Once "
        "an order has been packed, it can no longer be cancelled through the app, since "
        "the rider is dispatched immediately after packing given Zepto's quick-delivery "
        "model. If a packed order cannot be delivered due to a Zepto-side issue (for "
        "example, rider unavailability), the order is auto-cancelled and fully refunded "
        "without any cancellation fee."
    ),
    "doc_06": (
        "If an order arrives with damaged, spoiled, or missing items, customers must "
        "report it within 24 hours of delivery through the 'Report an Issue' button "
        "on the order page. Zepto ships a free replacement or issues a full refund for "
        "damaged, spoiled, or missing items without requiring the customer to return "
        "the original item, unless the order value exceeds INR 1000, in which case a "
        "photo of the issue must be submitted through the report form before a "
        "replacement or refund is processed."
    ),
    "doc_07": (
        "Zepto gift cards are available in fixed denominations of INR 100, INR 250, "
        "INR 500, and INR 1000, and are delivered by email or SMS within minutes of "
        "purchase. Gift cards are valid for 1 year from the date of issue and carry no "
        "maintenance fees. Gift card balance can be combined with one other payment "
        "method at checkout but cannot be combined with another gift card in the same "
        "transaction. Gift card balance cannot be redeemed for cash except where "
        "required by law."
    ),
    "doc_08": (
        "Zepto customer support is available via in-app chat 24 hours a day, 7 days a "
        "week, given the time-sensitive nature of quick commerce deliveries. Average "
        "in-app chat response time is under 2 minutes. Email support is also available "
        "for non-urgent queries and is answered within 24 hours on business days. Phone "
        "support is not offered."
    ),
}

# Structured prompt template: role, context, task, format, length, negative rule, example.
PROMPT_TEMPLATE = """
ROLE: You are Zepto's policy support assistant.
CONTEXT: Answer only from the policy excerpts supplied for this question.
TASK: Answer the customer's question accurately.
NEGATIVE CONSTRAINT: Do not use information that is not present in the supplied context.
FORMAT: Return JSON with answer (string), sources (list of document IDs),
and confidence (number from 0 to 1).
LENGTH: Keep the answer to 1–3 short sentences.

FEW-SHOT EXAMPLE
Context: {{"doc_02": "Perishable items may be reported within 24 hours."}}
Question: When should I report a damaged perishable item?
Answer: {{"answer": "Report it within 24 hours of delivery.",
"sources": ["doc_02"], "confidence": 1.0}}

Context: {context}
Question: {question}
"""


class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# Write all eight exact policy files.
DOCS_DIR.mkdir(parents=True, exist_ok=True)
for document_id, text in POLICY_DOCS.items():
    (DOCS_DIR / f"{document_id}.txt").write_text(text, encoding="utf-8")

# Load the local embedding model and persist the document embeddings in Chroma.
print("Loading local embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    document_ids = list(POLICY_DOCS.keys())
    document_texts = [POLICY_DOCS[doc_id] for doc_id in document_ids]
    document_embeddings = embedding_model.encode(
        document_texts,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=document_ids,
        documents=document_texts,
        embeddings=document_embeddings,
    )
    print(f"Indexed {len(document_ids)} policy documents in Chroma.")

print(f"Policy index contains {collection.count()} documents.")


def classify_intent(state: AssistantState) -> dict:
    query = state["query"]
    if not MOCK_LLM:
        raise NotImplementedError(
            "The optional real-LLM mode is not configured; use MOCK_LLM=1."
        )

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]
    query_lower = query.lower()
    intent = (
        "policy_question"
        if any(keyword in query_lower for keyword in policy_keywords)
        else "general_question"
    )
    return {"intent": intent}


def retrieve_and_answer(state: AssistantState) -> dict:
    query = state["query"]
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances"],
    )

    source_ids = results["ids"][0]
    retrieved_texts = results["documents"][0]

    if not MOCK_LLM:
        raise NotImplementedError(
            "The optional real-LLM mode is not configured; use MOCK_LLM=1."
        )

    top_chunk_snippet = retrieved_texts[0][:200]
    answer = f"Based on the retrieved context: {top_chunk_snippet}"

    # Constructing the prompt keeps the required template available for extension.
    _prompt = PROMPT_TEMPLATE.format(
        context="\n".join(retrieved_texts),
        question=query,
    )

    return {
        "answer": answer,
        "sources": source_ids,
        "confidence": 1.0,
    }


def direct_answer(state: AssistantState) -> dict:
    if not MOCK_LLM:
        raise NotImplementedError(
            "The optional real-LLM mode is not configured; use MOCK_LLM=1."
        )

    return {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0,
    }


def route_by_intent(state: AssistantState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


builder = StateGraph(AssistantState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)
builder.set_entry_point("classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)
assistant_graph = builder.compile()

app = FastAPI(title="Zepto Policy Support Assistant")


@app.get("/health")
def health():
    return {"status": "ok", "indexed_documents": collection.count()}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    result = assistant_graph.invoke({"query": request.query})
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )