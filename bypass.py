import os
import openai
import mysql.connector
from dotenv import load_dotenv

# .env 환경변수 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# MySQL 연결
def get_patent_summaries():
    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        port=os.getenv("MYSQL_PORT"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )
    cursor = connection.cursor()
    cursor.execute("SELECT id, patent_number, title, summary FROM patents LIMIT 5")
    results = cursor.fetchall()
    cursor.close()
    connection.close()
    return results  # (id, patent_number, title, summary)

# GPT로 우회 전략 생성
def generate_bypass_strategy(summary_text):
    system_prompt = (
        "당신은 특허 회피전략 전문가입니다. "
        "주어진 특허 요약 내용을 기반으로 법적 침해를 피하면서 기술적으로 동일한 목적을 달성할 수 있는 "
        "우회 전략을 제시하세요. 가능한 전략을 3가지 이상 설명하세요. "
        "내용은 창의적이고 실현 가능한 형태여야 하며, 각 전략은 간결하게 2~3문장으로 구성해 주세요."
    )

    user_prompt = f"다음은 특허 요약입니다:\n\n{summary_text}\n\n이 특허를 회피할 수 있는 전략을 제안해 주세요."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message['content']

# 메인 실행
if __name__ == "__main__":
    patents = get_patent_summaries()
    for pid, patent_number, title, summary in patents:
        print(f"\n🔎 특허 ID {pid} | 특허번호: {patent_number}")
        print(f"📌 제목: {title}")
        print(f"📄 요약: {summary}\n")
        strategy = generate_bypass_strategy(summary)
        print(f"🛡 우회 전략:\n{strategy}\n{'-'*60}")
