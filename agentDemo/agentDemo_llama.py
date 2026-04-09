from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)


# load documents(加载文档)
documents = SimpleDirectoryReader(input_files=["./rag_agent/窦茂川简历.pdf"]).load_data()

# split document(切分文档)
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# print(len(nodes))
# print(nodes[0].text)
# print(nodes[0].metadata)

# load model
Settings.llm = Ollama(model="llama3.2:1b")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 定义索引
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# define summary query engine和vector query engine
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

# define tools
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to the resume"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the resume."
    ),
)

# define router
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

response = query_engine.query("What is the summary of the document?")
# print(str(response))
print(len(response.source_nodes))
