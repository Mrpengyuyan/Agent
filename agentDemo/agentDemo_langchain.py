import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent

embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = ChatOllama(model="llama3.2:1b")

loader = PyPDFLoader("/Users/maochuandou/Internship/Agent/rag_agent/窦茂川简历.pdf")
# loader = DirectoryLoader("./文件夹/"，glob = “**/*.pdf”, loader_cls = PyPDFLoader)
docs = loader.load()

assert len(docs) == 1
# print(f"Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content[:500])

# 切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# print(f"Split docu post into {len(all_splits)} sub-documents.")

# print(type(docs))
# print(type(docs[0]))

def build_chunk_id(doc):
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    start_index = doc.metadata.get("start_index", "")
    raw = f"{source}|{page}|{start_index}|{doc.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


document_ids = [build_chunk_id(doc) for doc in all_splits]

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Rebuild this demo collection on each run so old random IDs do not accumulate.
# vector_store.reset_collection()
stored_ids = vector_store.add_documents(documents=all_splits, ids=document_ids)

# print(f"Upserted {len(stored_ids)} chunks.")
# print(stored_ids)


@tool
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

tools = [retrieve_context]

prompt = (
    "You have exactly one tool: retrieve_context. "
    "Use it when you need information from the resume. "
    "The tool accepts exactly one argument named 'query', and its value must be a plain string. "
    "Do not invent any other tool names or function schemas. "
    "If the resume does not contain the answer, say you don't know."
)


agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "Did Dou Maochuan receive the National Scholarship?\n"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
