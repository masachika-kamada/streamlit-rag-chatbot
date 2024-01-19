import streamlit as st
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS, Annoy, ScaNN
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    load_dotenv()
except Exception as e:
    print(f"Can't load API key : {e}")


def load_db():
    with open("vectorstore/store_type.txt", mode="r", encoding="utf-8") as f:
        store_type = f.read()
    if store_type == "FAISS":
        return FAISS.load_local("vectorstore")
    elif store_type == "Annoy":
        return Annoy.load_local("vectorstore", OpenAIEmbeddings())
    elif store_type == "ScaNN":
        return ScaNN.load_local("vectorstore", OpenAIEmbeddings())
    else:
        raise ValueError("Unsupported vector store type")


def init_page():
    st.set_page_config(page_title="RAG App", page_icon="🧑‍💻")
    st.header("RAG App")


def main():
    db = load_db()
    init_page()

    embeddings = OpenAIEmbeddings()
    model = "gpt-4-1106-preview"
    chain = load_qa_chain(ChatOpenAI(temperature=0.7, model_name=model), chain_type="stuff")

    # UI用の会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # UI用の会話履歴を表示
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    # ユーザーの入力を監視
    if user_input := st.chat_input("質問を入力して下さい"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("ChatGPT is typing ..."):
                embedding_vector = embeddings.embed_query(user_input)
                docs_and_scores = db.similarity_search_by_vector(embedding_vector)
                response = chain.invoke({"input_documents": docs_and_scores, "question": user_input}, return_only_outputs=True)
                st.markdown(response["output_text"])
        st.session_state.messages.append(AIMessage(content=response["output_text"]))


if __name__ == "__main__":
    main()
