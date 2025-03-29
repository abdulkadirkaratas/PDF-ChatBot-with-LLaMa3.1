import requests
import json
import panel as pn
import param
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

pn.extension()

# Embedding işlemi
def get_embedding(text, model):
	try:
		url = "http://localhost:11434/api/embeddings"
		headers = {"Content-Type": "application/json"}
		data = {"model": model, "prompt": text}
		response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
		response.raise_for_status()
		return response.json().get("embedding")
	except Exception as e:
		print(f"Embedding Error: {e}")
		return [0] * 768

# Embedding sınıfı
class OllamaEmbedding:
	def __init__(self, model):
		self.model = model

	def embed_documents(self, texts):
		return [get_embedding(text, self.model) for text in texts]

	def embed_query(self, text):
		return get_embedding(text, self.model)

# LLM - RAG
def load_db(file_path, chain_type="stuff", k=1):
	try:
		loader = PyPDFLoader(file_path)
		documents = loader.load()
		text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
		docs = text_splitter.split_documents(documents)
		embeddings = OllamaEmbedding("nomic-embed-text")
		db = DocArrayInMemorySearch.from_documents(docs, embeddings)
		retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
		return ConversationalRetrievalChain.from_llm(
			llm=OllamaLLM(model="llama3.1", temperature=0),
			chain_type=chain_type,
			retriever=retriever,
			return_source_documents=True,
			return_generated_question=True,
		)
	except Exception as e:
		print(f"Database Loading Error: {e}")
		return None

# Chatbot arayüzü
class ChatBotApp(param.Parameterized):
	chat_history = param.List([])  # [(human, ai), ...] formatında olacak
	pdf_path = param.String(default=None)

	def __init__(self, **params):
		super().__init__(**params)
		self.qa = None
		# ChatInterface'i oluştururken callback'i bağlıyoruz
		self.chat = pn.chat.ChatInterface(
			callback=self.ask_question,
			placeholder_text="...",
			styles={
				"background": "#D3D3D3",
				"color": "black"
			},
			callback_exception='verbose'  # Hata detaylarını göster
		)

	def load_pdf(self, event):
		if file_input.value:
			file_input.save("uploaded_pdf.pdf")
			self.pdf_path = "uploaded_pdf.pdf"
			self.qa = load_db(self.pdf_path)
			if self.qa is None:
				self.chat.send("An error occurred while loading the PDF. Please try again.", user="Assistant", respond=False)
				return
			pdf_viewer.object = self.pdf_path
			self.chat.clear()
			self.chat_history.clear()
			self.chat.send("PDF uploaded successfully! I'm waiting for your questions.", user="Assistant", respond=False)

	def ask_question(self, query, **kwargs):
		if not query.strip():
			return "Please ask a valid question."
		if not self.qa:
			return "Please ask a valid question."

		try:
			# Chat history formatını ConversationalRetrievalChain için uygun hale getir
			formatted_history = [(human, ai) for human, ai in self.chat_history if ai is not None]

			# Chatbot yanıtını al
			result = self.qa.invoke({"question": query, "chat_history": formatted_history})
			if result is None or "answer" not in result:
				response = "Sorry, an error occurred. No response was received."
			else:
				response = result["answer"]

			# Sohbet geçmişine ekle
			self.chat_history.append((query, response))
		except Exception as e:
			response = f"An error occurred: {str(e)}"
			self.chat_history.append((query, response))

		# Chatbot yanıtını döndür
		return response

# UI Bileşenleri
chat_app = ChatBotApp()
file_input = pn.widgets.FileInput(accept=".pdf", description="Upload PDF")
upload_button = pn.widgets.Button(name="Upload PDF", button_type="primary")
upload_button.on_click(chat_app.load_pdf)
pdf_viewer = pn.pane.PDF(object=None, width=635, height=500)

# Arayüz düzeni
dashboard = pn.template.MaterialTemplate(
	title="PDF ChatBot",
	theme="dark",
	header_background="#1E1E1E",
	sidebar_width=700,
	sidebar=[
		pn.Column(
			pn.pane.Markdown("## PDF Viewer", styles={"color": "white", "font-size": "20px"}),
			pn.Row(file_input, upload_button, align="center"),
			pdf_viewer,
			styles={"border-right": "4px solid grey"}
		)
	],
	main=[
		pn.Column(
			pn.pane.Markdown("## 💬 Chat with your PDF", styles={"color": "white", "font-size": "20px"}),
			chat_app.chat
		)
	]
)

pn.serve(dashboard, port=5006, show=True)
