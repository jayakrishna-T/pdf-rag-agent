from typing import TypedDict

from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import os
from retriever import retrieve_documents


class AgentState(TypedDict):

    question: str
    documents: list
    context: str
    recommendation: str
    final_response: str


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
    
)



def user_question_node(state):

    return {
        "question": state["question"]
    }


def retrieval_decision_node(state):

    question = state["question"]

    prompt = f"""
Determine whether this question requires
information from the PDF knowledge base.

Question:
{question}

Reply only with RETRIEVE.
"""

    llm.invoke(prompt)

    return {
        "documents": []
    }


def pdf_retrieval_node(state):

    question = state["question"]

    documents = retrieve_documents(question)

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    return {
        "documents": documents,
        "context": context
    }


def recommendation_generation_node(state):

    question = state["question"]

    context = state["context"]

    if not context.strip():

        return {
            "recommendation":
            "I could not find sufficient information in the provided PDFs."
        }

    prompt = f"""
You are a PDF-based recommendation assistant.

Answer ONLY using the PDF context below.

Never use outside knowledge.

If the answer cannot be supported by the
PDF context, say:

"I could not find sufficient information in the provided PDFs."

PDF CONTEXT:
{context}

QUESTION:
{question}

Give a concise recommendation.
"""

    response = llm.invoke(prompt)

    return {
        "recommendation": response.content
    }


def final_response_node(state):

    return {
        "final_response":
        state["recommendation"]
    }


graph_builder = StateGraph(AgentState)


graph_builder.add_node(
    "user_question",
    user_question_node
)

graph_builder.add_node(
    "retrieval_decision",
    retrieval_decision_node
)

graph_builder.add_node(
    "pdf_retrieval",
    pdf_retrieval_node
)

graph_builder.add_node(
    "recommendation_generation",
    recommendation_generation_node
)

graph_builder.add_node(
    "final_response",
    final_response_node
)


graph_builder.set_entry_point(
    "user_question"
)


graph_builder.add_edge(
    "user_question",
    "retrieval_decision"
)

graph_builder.add_edge(
    "retrieval_decision",
    "pdf_retrieval"
)

graph_builder.add_edge(
    "pdf_retrieval",
    "recommendation_generation"
)

graph_builder.add_edge(
    "recommendation_generation",
    "final_response"
)

graph_builder.add_edge(
    "final_response",
    END
)


graph = graph_builder.compile()