"""
Week 3, Days 14-15 — Ollama chatbot, grounded by RAG retrieval and the
patient's own report context.

Requires Ollama running locally (already installed) with the llama3.2
model pulled (`ollama pull llama3.2` — already done per the brief).

Standalone test (NOT executed here):
    python src/chatbot.py --disease kidney --question "what does high blood urea mean?"
"""
import argparse
from pathlib import Path

import ollama

import sys
sys.path.append(str(Path(__file__).parent))

from rag import retrieve

OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a medical-information assistant embedded in a
patient-facing healthcare monitoring demo. Follow these rules strictly:

1. You are NOT a doctor and must never give a diagnosis or treatment
   instructions. You explain what lab values and model predictions mean in
   plain language.
2. Base your answers ONLY on the "Reference material" provided in each
   prompt and the patient's own report context. If the reference material
   doesn't cover the question, say so plainly instead of guessing.
3. Always keep a calm, clear, non-alarming tone, even when values are
   abnormal.
4. End every response with a brief reminder to consult a qualified
   healthcare professional for actual diagnosis or treatment decisions.
"""


def format_patient_context(patient_context):
    """
    patient_context: dict like
      {"disease": "Chronic Kidney Disease", "values": {...},
       "prediction_label": "...", "confidence": 0.83}
    Returns a short plain-text block to embed in the prompt.
    """
    if not patient_context:
        return "No patient report has been submitted yet this session."

    lines = [
        f"Disease/report type: {patient_context.get('disease')}",
        f"Model prediction: {patient_context.get('prediction_label')}",
    ]
    confidence = patient_context.get("confidence")
    if confidence is not None:
        lines.append(f"Model confidence: {confidence:.1%}")
    values = patient_context.get("values", {})
    if values:
        lines.append("Reported values:")
        for k, v in values.items():
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


def format_reference_chunks(chunks):
    if not chunks:
        return "No matching reference material was found for this question."
    parts = []
    for c in chunks:
        parts.append(f"### {c['title']}\n{c['text']}")
    return "\n\n".join(parts)


def build_user_prompt(patient_context, reference_chunks, question):
    return f"""Patient report context:
{format_patient_context(patient_context)}

Reference material (use only this to answer factual questions about lab markers):
{format_reference_chunks(reference_chunks)}

Patient's question: {question}
"""


def _build_messages(question, disease_key, patient_context=None, chat_history=None, top_k=4):
    chunks = retrieve(question, disease=disease_key, top_k=top_k)
    user_prompt = build_user_prompt(patient_context, chunks, question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def answer_patient_question(question, disease_key, patient_context=None, chat_history=None, top_k=4):
    """
    Non-streaming version — waits for the full response. Kept for the CLI
    self-test below; the Streamlit app uses stream_patient_answer instead.
    """
    messages = _build_messages(question, disease_key, patient_context, chat_history, top_k)
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]


def stream_patient_answer(question, disease_key, patient_context=None, chat_history=None, top_k=4):
    """
    Streaming version — yields response text incrementally, chunk by chunk,
    as Ollama generates it. Use with st.write_stream(...) in Streamlit, or
    iterate it directly (e.g. `for piece in stream_patient_answer(...): ...`).
    """
    messages = _build_messages(question, disease_key, patient_context, chat_history, top_k)
    for chunk in ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the RAG-grounded Ollama chatbot")
    parser.add_argument("--disease", required=True, choices=["kidney", "liver"])
    parser.add_argument("--question", required=True)
    parser.add_argument("--stream", action="store_true", help="Print the response as it streams in")
    args = parser.parse_args()

    try:
        if args.stream:
            print("\nAssistant: ", end="", flush=True)
            for piece in stream_patient_answer(args.question, args.disease):
                print(piece, end="", flush=True)
            print()
        else:
            answer = answer_patient_question(args.question, args.disease)
            print("\nAssistant:\n" + answer)
    except Exception as e:
        print(f"Error calling Ollama — is it running? (`ollama serve` / check it's installed): {e}")
