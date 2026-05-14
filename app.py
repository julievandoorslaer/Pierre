from flask import Flask, render_template, request, Response, stream_with_context
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os
import re
import datetime
from src.prompt import system_prompt

# Tag aan begin van antwoord: [BRON] = context gebruikt, [GEEN] = smalltalk/onbekend.
# Tolerant voor casing, kleine variaties, optionele spaties ervoor.
TAG_PATTERN = re.compile(r'^\s*\[\s*(BRON|GEEN)\s*\]\s*', re.IGNORECASE)
# Aantal chars dat we maximaal bufferen om de tag te vinden voor we opgeven.
TAG_BUFFER_LIMIT = 40
# Cosine similarity drempel: bij ontbrekende tag valt beslissing terug op deze score.
# Tunen op basis van logs; 0.5 is een conservatief startpunt voor multilingual-e5-small.
SOURCE_SCORE_THRESHOLD = 0.5


app = Flask(__name__)
load_dotenv()

# Setup
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
embeddings = download_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name="medical-chatbot", embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})

chatModel = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.getenv('OLLAMA_API_KEY')},
    streaming=True 
)

# Per-sessie chatgeschiedenis. Wordt later via RunnableWithMessageHistory ingelezen.
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def log_interaction(question, answer):
    try:
        with open("chat_logs.txt", "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- {timestamp} ---\nPatiënt: {question}\nPierre: {answer}\n{'-'*50}\n\n")
    except Exception as e: print(f"Log error: {e}")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 1. De "Herschrijver" (Vertaalt 'hiervan' naar de echte context) ---
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Je bent een medische terminologie-expert. Je krijgt een chatgeschiedenis en een nieuwe gebruikersvraag. "
        "Patiënten maken vaak typefouten in complexe termen.\n\n" # (bijv. 'ecris' ipv 'ECIRS', 'protsaat' ipv 'prostaat', 'urologie' ipv 'urologie')
        "JOUW TAAK:\n"
        "1. Corrigeer eventuele spelfouten in de vraag.\n"
        "2. Gebruik de chatgeschiedenis om context toe te voegen (vertaal 'hiervan' of 'die operatie' naar de juiste term).\n"
        "3. Formuleer een beknopte, foutloze zoekopdracht voor een medische database.\n\n"
        "Antwoord ENKEL met de gecorrigeerde zoekopdracht, geen extra uitleg."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Dit is een klein hulp-ketentje
question_generator = contextualize_prompt | chatModel | StrOutputParser()

# --- 2. De Slimme Zoekfunctie (Retriever) ---
def branched_retriever(inputs):
    # Als er al wat gezegd is, laat Pierre de vraag verduidelijken voor de PDF-zoeker
    if inputs.get("chat_history") and len(inputs["chat_history"]) > 0:
        optimized_query = question_generator.invoke(inputs)
        return retriever.invoke(optimized_query)
    # Eerste vraag? Zoek gewoon direct
    return retriever.invoke(inputs["input"])

# --- 3. De Hoofd Chain (de 'Pierre' die de patiënt ziet) ---
pierre_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rag_chain = (
    {
        "context": RunnableLambda(branched_retriever) | RunnableLambda(format_docs),
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"]
    }
    | pierre_prompt
    | chatModel
    | StrOutputParser()
)

# --- 4. Finale wrapper: voegt automatisch chatgeschiedenis toe per session_id ---
with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def _decide_use_sources(tag_decision, top_score):
    """Beslis of we de bronnenlijst tonen.

    Tag heeft voorrang ([BRON]=True, [GEEN]=False); bij ontbrekende tag
    valt het terug op de cosine-similarity score van de top-match.
    """
    if tag_decision is True:
        return True
    if tag_decision is False:
        return False
    return top_score >= SOURCE_SCORE_THRESHOLD

@app.route("/")
def home(): return render_template("home.html")

@app.route("/chat")
def index(): return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    if not msg: return ""

    # Bronnen + scores ophalen. Score is fallback als de [BRON]/[GEEN]-tag in het
    # antwoord ontbreekt of niet leesbaar is.
    docs_with_scores = docsearch.similarity_search_with_score(msg, k=6)
    sources = list(set([
        os.path.basename(doc.metadata.get('source', 'Onbekende bron'))
        for doc, _ in docs_with_scores
    ]))
    top_score = max((score for _, score in docs_with_scores), default=0.0)
    source_text = "\n\n<br><strong>Bronnen:</strong> " + ", ".join(sources)

    def generate():
        full_response = ""
        buffer = ""
        tag_decision = None  # True = [BRON], False = [GEEN], None = tag niet gevonden
        tag_resolved = False

        for chunk in with_message_history.stream(
            {"input": msg},
            config={"configurable": {"session_id": "temp_user"}}
        ):
            if not tag_resolved:
                buffer += chunk
                match = TAG_PATTERN.match(buffer)
                if match:
                    tag_decision = match.group(1).upper() == "BRON"
                    remainder = buffer[match.end():]
                    tag_resolved = True
                    if remainder:
                        full_response += remainder
                        yield remainder
                    buffer = ""
                elif len(buffer) >= TAG_BUFFER_LIMIT:
                    # Tag niet gevonden in eerste chars: model heeft hem vergeten.
                    # Stuur buffer alsnog door, beslissing valt terug op score.
                    tag_resolved = True
                    full_response += buffer
                    yield buffer
                    buffer = ""
            else:
                full_response += chunk
                yield chunk

        # Stream eindigde voordat buffer-limit bereikt was: flush wat er is.
        if buffer:
            stripped = TAG_PATTERN.sub("", buffer, count=1)
            if stripped != buffer:
                tag_decision = "[BRON]" in buffer.upper()
            full_response += stripped
            yield stripped

        if _decide_use_sources(tag_decision, top_score):
            yield source_text
            full_response += source_text

        log_interaction(msg, full_response)

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)