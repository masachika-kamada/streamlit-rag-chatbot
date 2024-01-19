from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


try:
    load_dotenv()
except Exception as e:
    print(f"Can't load API key : {e}")

models = ["gpt-3.5-turbo", "gpt-4-1106-preview"]
llm = ChatOpenAI(temperature=0.7, model=models[1])

""" 1. Zero-shot """
# ChatGPTに渡すPromptのテンプレートを定義する
prompt = ChatPromptTemplate.from_messages(
    ("human", "次の入力をInstagram風の構文に変換して下さい: {text}")
)
# PromptやLLMを連携するためのChainを定義する
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke("おはよう！")["text"])

""" 2. Few-shot (in-context learning) """
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたはInstagramを頻繁に利用する10代の女性です。与えられた文章をInstagramでよく利用される構文に変換します。"),
        ("human", "おはよう"),
        ("ai", "おはようございます🌞 新しい一日が始まったよ！#おはよう #新しい一日 #朝"),
        ("human", "{text}"),
    ]
)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke("こんばんは")["text"])


# 出力結果
# ----------------------------------------
# Instagramでは、ハッシュタグや絵文字を使ってメッセージを飾ることが一般的です。したがって、「おはよう！」というメッセージをInstagram風にすると、以下のようになります：

# おはよう！☀️😊 #おはようございます #朝 #新しい一日 #ポジティブスタート
# こんばんは✨ みんないい夜を過ごしてる？🌙💕 #こんばんは #夜更かしクラブ #ゆっくり時間
