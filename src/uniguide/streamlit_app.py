from __future__ import annotations

import streamlit as st

from uniguide.config import Settings
from uniguide.foundry import FoundryLocalRuntime
from uniguide.rag import RagService


st.set_page_config(
    page_title="UniGuide Local RAG",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def get_service() -> RagService:
    settings = Settings()
    runtime = FoundryLocalRuntime(
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )
    return RagService(settings=settings, runtime=runtime)


service = get_service()

st.title("🎓 UniGuide Local RAG")
st.caption(
    "Haliç Üniversitesi belgelerinde yerel, kaynak gösteren ve çevrimdışı soru-cevap"
)
st.info(
    "Yanıtlar yalnızca yüklenen belgelerden üretilir. Kesin akademik işlemler için "
    "üniversitenin güncel resmî duyurularını kontrol edin."
)

with st.sidebar:
    st.header("Bilgi tabanı")
    document_count, chunk_count = service.database.stats()
    st.metric("Belge", document_count)
    st.metric("Chunk", chunk_count)
    rebuild = st.checkbox("İndeksi sıfırdan oluştur")
    if st.button("Belgeleri indeksle", type="primary", use_container_width=True):
        status = st.status("İndeks hazırlanıyor...", expanded=True)

        def show_progress(message: str) -> None:
            status.write(message)

        try:
            report = service.index_documents(rebuild=rebuild, progress=show_progress)
            status.update(
                label=(
                    f"Tamamlandı: {report.indexed_documents} belge, "
                    f"{report.indexed_chunks} chunk"
                ),
                state="complete",
            )
            st.rerun()
        except Exception as exc:
            status.update(label="İndeksleme başarısız", state="error")
            st.error(str(exc))

    st.divider()
    st.markdown(
        "**Modeller**\n\n"
        f"Embedding: `{service.settings.embedding_model}`\n\n"
        f"Chat: `{service.settings.chat_model}`"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            with st.expander(
                f"{source['citation']} · benzerlik {source['score']:.3f}"
            ):
                st.write(source["content"])

question = st.chat_input("Örneğin: Yandal programına ne zaman başvurabilirim?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Yerel belgelerde aranıyor..."):
                result = service.ask(question)
            st.markdown(result.answer)
            serialized_sources = []
            for source in result.sources:
                with st.expander(
                    f"{source.citation} · benzerlik {source.score:.3f}"
                ):
                    st.write(source.content)
                serialized_sources.append(
                    {
                        "citation": source.citation,
                        "score": source.score,
                        "content": source.content,
                    }
                )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": serialized_sources,
                }
            )
        except Exception as exc:
            st.error(str(exc))
