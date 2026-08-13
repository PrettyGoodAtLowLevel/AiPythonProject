#start
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import chromadb.utils.embedding_functions as ef

#load memory thing
load_dotenv()
db = chromadb.PersistentClient(path = "./chroma_db")
memories = db.get_or_create_collection("my_facts")

def add_memory(memories, added_doc):
    memories.upsert(documents= [added_doc],
    ids = [f"fact{memories.count() + 1}"])

#get basic question and best memories based on it
basicQuestion = str(input("Enter Your Question"))
results = memories.query(query_texts = [basicQuestion], n_results=3)

#make big question out of memories
mems = results["documents"][0]
memory_text = " ".join(mems)
finalQuestion = memory_text + " " + basicQuestion

#create ai
client = OpenAI(base_url= "https://api.groq.com/openai/v1",
api_key=os.getenv("GITHUB_TOKEN"),)

#ask ai big question and print message
add_memory(memories, f"previous conversation question: {finalQuestion}")
r = client.chat.completions.create(model="llama-3.3-70b-versatile",
messages=[{"role": "user", "content": finalQuestion}],)
print(f"\n{r.choices[0].message.content}")