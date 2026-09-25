import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Q&A Assistant")

st.write(
    "Upload PDF or TXT documents and ask questions "
    "based only on their content."
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# EXTRACT TEXT
# ==========================================

def extract_text(file):

    if file.name.lower().endswith(".pdf"):

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    elif file.name.lower().endswith(".txt"):

        return file.read().decode("utf-8")

    return ""


# ==========================================
# CREATE DOCUMENT CHUNKS
# ==========================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==========================================
# RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve_chunks(
    question,
    chunks,
    embeddings,
    model,
    top_k=3
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    similarity_scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    ranked_indices = similarity_scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "text": chunks[index],
                "score": float(
                    similarity_scores[index]
                ),
                "index": int(index)
            }
        )

    return results


# ==========================================
# UPLOAD FILES
# ==========================================

uploaded_files = st.file_uploader(
    "Upload your PDF or TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# ==========================================
# PROCESS DOCUMENTS
# ==========================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} document(s) uploaded."
    )

    all_chunks = []
    chunk_sources = []


    # ======================================
    # TEXT EXTRACTION + CHUNKING
    # ======================================

    for uploaded_file in uploaded_files:

        st.subheader(
            f"📄 {uploaded_file.name}"
        )

        try:

            text = extract_text(uploaded_file)

        except Exception as e:

            st.error(
                f"Could not read {uploaded_file.name}"
            )

            st.exception(e)

            continue


        if not text.strip():

            st.warning(
                f"No readable text was found in "
                f"{uploaded_file.name}."
            )

            continue


        st.write(
            f"Extracted {len(text)} characters."
        )


        chunks = create_chunks(text)


        if not chunks:

            st.warning(
                f"No chunks were created from "
                f"{uploaded_file.name}."
            )

            continue


        st.write(
            f"Created {len(chunks)} chunks."
        )


        all_chunks.extend(chunks)


        for chunk in chunks:

            chunk_sources.append(
                uploaded_file.name
            )


        # Show extracted text

        with st.expander(
            "View extracted text"
        ):

            st.text(
                text[:5000]
            )


        # Show chunks

        with st.expander(
            "View document chunks"
        ):

            for i, chunk in enumerate(chunks):

                st.markdown(
                    f"**Chunk {i + 1}**"
                )

                st.text(chunk)


    # ======================================
    # GENERATE EMBEDDINGS
    # ======================================

    if all_chunks:

        st.subheader(
            "🔢 Document Embeddings"
        )


        try:

            with st.spinner(
                "Loading embedding model..."
            ):

                model = load_model()


            with st.spinner(
                "Generating embeddings..."
            ):

                embeddings = model.encode(
                    all_chunks,
                    convert_to_numpy=True
                )


            st.success(
                "Embeddings generated successfully."
            )


            st.write(
                f"Number of chunks: "
                f"{len(all_chunks)}"
            )


            st.write(
                f"Embedding dimensions: "
                f"{embeddings.shape[1]}"
            )


            st.write(
                f"Embedding matrix shape: "
                f"{embeddings.shape}"
            )


        except Exception as e:

            st.error(
                "An error occurred while "
                "generating embeddings."
            )

            st.exception(e)

            st.stop()


        # ==================================
        # QUESTION INPUT
        # ==================================

        st.subheader(
            "💬 Ask a Question"
        )


        question = st.text_input(
            "Ask a question about your documents:"
        )


        # ==================================
        # EMPTY QUESTION CHECK
        # ==================================

        if question.strip():


            # ==================================
            # SIMILARITY RETRIEVAL
            # ==================================

            with st.spinner(
                "Searching your documents..."
            ):

                results = retrieve_chunks(
                    question,
                    all_chunks,
                    embeddings,
                    model,
                    top_k=3
                )


            # ==================================
            # GROUNDING THRESHOLD
            # ==================================

            MIN_SIMILARITY = 0.20

            best_score = results[0]["score"]


            # ==================================
            # RETRIEVED CONTEXT
            # ==================================

            st.subheader(
                "🔎 Retrieved Context"
            )


            for i, result in enumerate(results):

                source = chunk_sources[
                    result["index"]
                ]


                st.markdown(
                    f"### Result {i + 1}"
                )


                st.write(
                    f"**Source:** {source}"
                )


                st.write(
                    f"**Similarity score:** "
                    f"{result['score']:.4f}"
                )


                st.info(
                    result["text"]
                )


            # ==================================
            # CHECK RELEVANCE
            # ==================================

            if best_score < MIN_SIMILARITY:

                st.warning(
                    "I could not find enough relevant "
                    "information in the uploaded documents "
                    "to answer this question."
                )

                st.info(
                    "Try asking a question that is "
                    "directly related to the uploaded "
                    "documents."
                )

                st.stop()


            # ==================================
            # GROUNDED ANSWER
            # ==================================

            st.subheader(
                "💡 Answer"
            )


            best_context = results[0]["text"]


            st.success(
                "Answer based on the most relevant "
                "retrieved document content:"
            )


            st.write(
                best_context
            )


            # ==================================
            # SOURCE
            # ==================================

            st.subheader(
                "📚 Source"
            )


            st.write(
                chunk_sources[
                    results[0]["index"]
                ]
            )


        else:

            st.info(
                "Enter a question above to search "
                "your uploaded documents."


            )

    else:

        st.warning(
            "No readable document content was found. "
            "Please upload a text-based PDF or TXT file."
        )


else:

    st.info(
        "Please upload a PDF or TXT document "
        "to begin."
    )