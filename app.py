import streamlit as st
from graph import graph


# ---------- Page setup ----------
st.set_page_config(
    page_title="PDF Recommendation Assistant",
    page_icon="📄",
    layout="centered",
)
 
st.title("📄 PDF Recommendation Assistant")
st.caption("Ask a question and I'll answer strictly from your PDF knowledge base.")


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "documents": [...]}


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This assistant retrieves relevant chunks from your PDFs "
        "(via FAISS + HuggingFace embeddings) and generates an answer "
        "using Groq's Llama 3.3 70B model, grounded only in that context."
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- Render chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("documents"):
            with st.expander(f"📚 Sources ({len(msg['documents'])})"):
                for i, doc in enumerate(msg["documents"], start=1):
                    source = doc.get("source", "Unknown")
                    page = doc.get("page", "N/A")
                    st.markdown(f"**Source {i}** — `{source}` (page {page})")
                    st.text(doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""))


# ---------- Chat input ----------
question = st.chat_input("Ask your question about the PDFs...")

if question:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run the LangGraph pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs and generating answer..."):
            try:
                result = graph.invoke(
                    {
                        "question": question,
                        "documents": [],
                        "context": "",
                        "recommendation": "",
                        "final_response": "",
                    }
                )
                answer = result.get("final_response", "Something went wrong — no response generated.")

                # Extract lightweight source info for display (avoids storing raw LC objects in session_state)
                raw_docs = result.get("documents", []) or []
                sources = [
                    {
                        "source": getattr(d, "metadata", {}).get("source", "Unknown"),
                        "page": getattr(d, "metadata", {}).get("page", "N/A"),
                        "content": getattr(d, "page_content", ""),
                    }
                    for d in raw_docs
                ]

            except Exception as e:
                answer = f"⚠️ An error occurred: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for i, doc in enumerate(sources, start=1):
                    st.markdown(f"**Source {i}** — `{doc['source']}` (page {doc['page']})")
                    st.text(doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "documents": sources}
    )
