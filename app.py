#start
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from doc_helper import read_file
import tempfile

#load env and get api key
load_dotenv()
key = os.getenv("GITHUB_TOKEN")

#load chroma db and memories + notes
DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")
db = chromadb.PersistentClient(path = DB_PATH)
brain = db.get_or_create_collection("documents")
memory = db.get_or_create_collection("conversations")

#chunks texts so that AI has better time understanding it
def chunk_it(text, size=800):
    bits = text.split(". ")
    chunks, current = [], ""
    for bit in bits:
        if len(current) + len(bit) < size:
            current += bit + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = bit + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks

#stores a document to the documents file
def store_document(file):
    chunks = chunk_it(read_file(file))
    prefix = file.name.replace(" ", "_")

    brain.upsert(documents=chunks,
    ids=[f"{prefix}_{i}" for i in range(len(chunks))])

    return len(chunks)

#stores a conversation to the conversations file
def store_conversation(question, answer):
    text = f"Q{question}\n A{answer}"
    chunks = chunk_it(text)
    turn = memory.count()
    memory.upsert(
        documents=[f"[past chat] {c}" for c in chunks],
        metadatas=[{"kind": "chat", "turn": turn} for c in chunks],
        ids = [f"turn{turn}_{i}" for i in range(len(chunks))],
    )
    return len(chunks)

#applies the streamlit theme
def apply_theme():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #000505;
            color: white;
            border-radius: 999px;
        }

        .stSidebar {
            background-color: #5D5D81;
        } </style> """, unsafe_allow_html=True)

#initialize
st.title("Craft.AI")
apply_theme()

if "messages" not in st.session_state:
    st.session_state.messages = []

#sidebar settings
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Enter your name:")

    preset = st.selectbox("Choose your color preset",
    ["Redstoner", "PvP Master", "Architect", "Adventurer", "Zany"])
    message_history = st.slider("Message History", 1, 15, 5)
    n_chunks = st.slider("Number of Chunks", 1, 15, 5)
    recall = st.slider("Number of Chunks for recall", 0, 15, 10)
    creativity = st.slider("Choose your creativity", 0.0, 1.0, 0.5)

    model = st.selectbox("Model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"], index=1)
    stream_it = st.checkbox("Stream the answer", value=True)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Clear History"):
        db.delete_collection("documents")
        st.rerun()
    if st.button("Clear all past chat history"):
        db.delete_collection("conversations")
        st.rerun()
    st.caption(f"{len(st.session_state.messages)} messages have been sent in this chat")
    st.caption(f"{brain.count()} chunks stored")
    st.caption(f"{memory.count()} conversations stored")

    if st.button("Save Settings"):
        st.write(f"Saved Settings, my name is {name}, your preset is {preset}, and your creativity is {creativity}")

#system prompt, prompt engineer to your desire
SYSTEM_PROMPT = (f"You are a Minecraft AI, you are very good at minecraft."
                 "You Must only discuss things related to Minecraft."
                 "You can give links and community resources related to Minecraft."
                 "You can also always give ideas to the user on the topic they are asking about,"
                 "If there is no topic, give ideas that could be related to "
                 "building, challenges, or redstone contraptions to make."
                 "You can make funny jokes as you wish, just not to be too mean."
                 "You must also fact check every single piece of information you are about to give"
                 "on any external website that is distinctly known for minecraft data, mainly use the Minecraft Wiki."
                 "Do Not discuss anything anything related to any other game"
                 "Or Work environment. If the user asks something that is not about minecraft"
                 "give a quick sorry message, and explain that you are a Minecraft AI and only talk things related to minecraft."
                 "All of the above are critical. Follow them closely.")

#load old messages
for old in st.session_state.messages:
    with st.chat_message(old["role"]):
        st.markdown(old["content"])

#get user input or file input
user_input = st.chat_input("Ask something in here", accept_file=True, file_type=["pdf", "txt"])

#parse file
if user_input:
    prompt = user_input.text
    if user_input.files:
        with st.spinner(f"Processing {user_input.files[0].name}.."):
            n = store_document(user_input.files[0])
            st.success(f"Stored {n} chunks inside of the chat, from {user_input.files[0].name}")

#respond to text
if user_input:
    #get basic prompt with AI settings
    prompt = user_input.text
    st.session_state.messages.append({"role": "user", "content": prompt})
    extendedPrompt = f"My name is: {name}, you, as the ai, your preset is {preset}, your creativity is {creativity}/1.0, prompt is {prompt}"
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key or st.secrets["GITHUB_TOKEN"],)

    with st.chat_message("user"):
        st.write(prompt)
    #check documents that relate to prompt
    notes = ""
    if brain.count() > 0:
        hits = brain.query(query_texts =[prompt], n_results = n_chunks)
        notes = "\n\n".join(hits["documents"][0])

        with st.expander("What I looked up"):
            for doc, dist in zip(hits["documents"][0], hits["distances"][0]):
                st.text(f"{dist:.3f}, {doc[:70]}")

    #check past conversations that relate to prompt
    recalled = ""
    if recall > 0 and memory.count() > message_history:
        old = memory.query(query_texts =[prompt], n_results = recall)
        recalled = "\n\n".join(old["documents"][0])

        with st.expander("What I remembered from past conversations"):
            for doc, dist  in zip(old["documents"][0], old["distances"][0]):
                st.text(f"{dist:.3f}, {doc[:70]}")

    #generate full complete prompt
    if notes or recalled:
        full_prompt = (f"These are POTENTIALLY, relevant notes to the users interest:\n {notes}\n"
        f"These are POTENTIALLY, relevant memories of past chats to the users interest:\n {recalled}"
        f"\nYou must answer this question with those notes and memories:\n {extendedPrompt}")
    else:
        full_prompt = extendedPrompt

    #create AI and stream answer
    with st.chat_message("assistant"):
        r = client.chat.completions.create(model=model, temperature=creativity,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
        + st.session_state.messages[-message_history: -1] +
        [{"role":"user", "content":full_prompt}],
        stream=True)

        #display answer
        thinking = st.expander("Thinking", expanded=True).empty()
        answer = st.empty()
        t = a = ""
        for chunk in r:
            d = chunk.choices[0].delta
            if getattr(d, "reasoning", None):
                t += d.reasoning
                thinking.markdown(f"*{t}*")
            if d.content:
                a += d.content
                answer.markdown(a)
    #save answer
    st.session_state.messages.append({"role": "assistant", "content": a})
    store_conversation(prompt, a)