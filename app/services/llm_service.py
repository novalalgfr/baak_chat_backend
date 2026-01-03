from app.core.state import resources
from app.utils.prompts import SYSTEM_PROMPT

def generate_answer(user_query: str, context_docs: list):
    groq_client = resources.get('groq_client')
    if not groq_client:
        return "Maaf, server AI sedang tidak siap."

    # Bangun Context String
    context_text = ""
    for doc in context_docs:
        label = doc['kategori']
        if doc['topik']: label += f" - {doc['topik']}"
        context_text += f"[{label}]\n{doc['content']}\n\n"

    final_prompt = SYSTEM_PROMPT + context_text

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0, 
            max_tokens=2048
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error LLM: {e}")
        return "Maaf, sedang terjadi gangguan koneksi ke server AI."