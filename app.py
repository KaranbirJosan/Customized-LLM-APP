import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
import os

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("EDM_Car_Parts_Catalog.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = ("You are a knowledgeable EDM car parts advisor. You provide accurate and concise advice on selecting parts for EDM cars. "
                      "You use the EDM Car Parts Catalog to provide information on various parts and their compatibility.")
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown("🚗 **Custom Part Picker for EDM Cars**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on the EDM Car Parts Catalog that is publicly available. "
        "We are not professional car mechanics, and the use of this chatbot is at your own risk. For professional advice, please consult a qualified mechanic.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["What parts do I need for an engine upgrade?"],
            ["Can you suggest parts for suspension tuning?"],
            ["What are the best tires for performance driving?"],
            ["How do I improve my car's braking system?"],
            ["What parts are compatible with my EDM car model?"],
            ["Can you help me with selecting exhaust systems?"],
            ["What are the recommended air filters for better performance?"],
            ["How do I enhance my car's aerodynamics?"]
        ],
        title='Custom Part Picker for EDM Cars 🚗'
    )

if __name__ == "__main__":
    demo.launch()

