import os

import requests
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def save_webpage_content(response_text, filepath="data/webpage.html"):
    """ウェブページの内容をファイルに保存する"""
    os.makedirs("data", exist_ok=True)
    with open(filepath, mode="w", encoding="utf-8") as file:
        file.write(response_text)


def load_and_transform_document(filepath):
    """HTMLドキュメントを読み込み、テキストに変換する"""
    loader = BSHTMLLoader(file_path=filepath, open_encoding="utf-8")
    datasource = loader.load()
    return datasource[0].page_content


def split_text_to_documents(text):
    """テキストをドキュメントに分割する"""
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
    )
    return text_splitter.create_documents([text])


def load_embeddings():
    """OpenAIの埋め込みモデルをロードする"""
    try:
        load_dotenv()
    except Exception as e:
        print(f"APIキーの読み込みに失敗しました: {e}")
        raise
    return OpenAIEmbeddings()


def create_vector_store(docs, embeddings):
    """ドキュメントのベクトルストアを作成する"""
    # Vector stores: ベクトルデータを管理するための機能
    # ここではFaissと呼ばれるベクトルDBを利用する
    # Faiss: Facebookが開発した効率的な近似最近傍検索ライブラリ
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_store")
    return db


def perform_question_answering(db, embeddings, query):
    """質問に答える処理を実行する"""
    # Q&A専用のChainを準備
    # `chain_type="stuff"` でpromptの内容を全て一度にAPIに詰め込んでいるが
    # map_reduce / refine などの方式も提供されいてる
    chain = load_qa_chain(ChatOpenAI(temperature=0.7), chain_type="stuff")
    embedding_vector = embeddings.embed_query(query)
    docs_and_scores = db.similarity_search_by_vector(embedding_vector)

    return chain.invoke(
        {"input_documents": docs_and_scores, "question": query},
        return_only_outputs=True,
    )


def fetch_webpage(url, headers):
    """ウェブページの内容を取得する"""
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"ウェブページの取得に失敗しました。ステータスコード: {response.status_code}")
    return response.text


def process_webpage(url):
    """ウェブページを処理するメイン関数"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.80"
    }
    webpage_content = fetch_webpage(url, headers)
    save_webpage_content(webpage_content)
    text_content = load_and_transform_document("data/webpage.html")
    documents = split_text_to_documents(text_content)
    embeddings = load_embeddings()
    db = create_vector_store(documents, embeddings)
    answer = perform_question_answering(db, embeddings, "どんなクラブがある？")
    print(answer)


if __name__ == "__main__":
    url = "https://www.microsoft.com/ja-jp/mscorp/mid-career/benefit"
    process_webpage(url)


# 実行結果
# ----------------------------------------
# {'output_text': '以下のクラブがあります：\n\n- 華道\n- クライミング\n- サーフィン\n- ヨガ\n- 剣道\n- 軽音楽\n- スノー\n- マラソン\n- 農業\n- バスケットボール\n- ファイトクラブ\n- e-sport\n- 数学\n\n他にもさまざまなクラブがありますので、詳細はリンク先をご確認ください。'}
