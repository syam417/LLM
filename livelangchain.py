from ollama import chat
from ollama import ChatResponse
from gtts import gTTS
import os
from playsound import playsound
import re
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms.base import LLM
from pydantic import BaseModel

# Wrapper untuk Model Ollama agar kompatibel dengan LangChain
class OllamaLLM(LLM, BaseModel):
    model_name: str = "llamania"  # Tambahkan atribut model_name dengan nilai default

    def _call(self, prompt: str, stop=None):
        try:
            response: ChatResponse = chat(model=self.model_name, messages=[{
                'role': 'user',
                'content': prompt,
            }])
            return response.message.content
        except Exception as e:
            return f"Kesalahan saat menghubungi model Ollama: {e}"

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self):
        return "ollama"

# Inisialisasi LangChain dengan Memory
def init_langchain():
    ollama_llm = OllamaLLM(model_name="llamania")  # Gunakan model lokal Ollama
    memory = ConversationBufferMemory()  # Memory untuk menyimpan konteks
    return ConversationChain(llm=ollama_llm, memory=memory)

# Fungsi untuk konversi teks ke suara
def text_to_speech(text):
    try:
        # Hapus teks dalam format markdown (* *)
        text_cleaned = re.sub(r'\*.*?\*', '', text)
        print(f"Mencoba membaca teks: {text_cleaned}")
        tts = gTTS(text=text_cleaned, lang='id')  # Menggunakan bahasa Indonesia
        tts.save("output.mp3")
        playsound("output.mp3")
        os.remove("output.mp3")  # Menghapus file setelah diputar
        print("Pembacaan selesai.")
    except Exception as e:
        print(f"Kesalahan saat memproses text-to-speech: {e}")

# Fungsi utama
def main():
    print("Selamat datang di AI Companion! Ketik 'keluar' untuk mengakhiri.")
    chat_chain = init_langchain()

    while True:
        # Ambil input pengguna
        user_input = input("Anda: ")
        if user_input.lower() == "keluar":
            print("Sampai jumpa!")
            text_to_speech("Sampai jumpa! Semoga harimu menyenangkan.")
            break

        # Kirim input pengguna ke LangChain
        try:
            response = chat_chain.run(user_input)
            # Tampilkan dan bacakan jawaban
            print(f"AI: {response}")
            text_to_speech(response)
        except Exception as e:
            print(f"Kesalahan saat memproses respons AI: {e}")

if __name__ == "__main__":
    main()
