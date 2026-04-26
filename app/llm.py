import os
from typing import List, Dict
from groq import Groq


class LLMClient:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate_answer(
        self,
        question: str,
        context: str,
        chat_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate answer grounded in retrieved context.
        Supports multi-turn conversation via chat_history.

        Args:
            question: Current user question
            context: Retrieved document chunks joined as string
            chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts
        """

        system_instruction = """
You are an expert research assistant specialized in analyzing technical and academic documents.
Your goal is to provide accurate, detailed answers grounded ONLY in the provided context.

RULES:
1. Base your answer strictly on the TECHNICAL CONTEXT provided.
2. If relevant, use bullet points or numbered lists for clarity.
3. If you can logically infer from metrics or descriptions, state your inference clearly.
4. If the context lacks relevant information, say exactly:
   "The provided document segments do not contain enough information to answer this question."
5. Do NOT use outside knowledge beyond what is in the context.
6. Take into account the conversation history to handle follow-up questions naturally.
"""

        # Build conversation turns for multi-turn support
        history_text = ""
        if chat_history:
            for turn in chat_history[-6:]:  # last 3 exchanges = 6 turns
                role = "User" if turn["role"] == "user" else "Assistant"
                history_text += f"{role}: {turn['content']}\n"

        user_prompt = f"""
TECHNICAL CONTEXT (retrieved document segments):
{context}

{"CONVERSATION HISTORY:" + chr(10) + history_text if history_text else ""}

CURRENT QUESTION:
{question}

Detailed Response:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.15,
                max_tokens=1500,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"An error occurred while generating the answer: {str(e)}"
