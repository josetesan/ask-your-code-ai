import dotenv
import torch
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
    load_index_from_storage,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import (
    CodeSplitter
)
from langchain_text_splitters import (
  RecursiveCharacterTextSplitter
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from sentence_transformers import SentenceTransformer
from IPython.display import Markdown, display
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM
)
from dotenv import load_dotenv
import logging


SOURCE_CODE = '/home/josete/src/spring-fault-tolerance'
# MODEL_URL = "https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q8_0.gguf"
#MODEL_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q6_K.gguf"
MODEL_URL= "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/raw/main/llama-2-7b-chat.Q4_K_M.gguf"


class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text, extra_info=extra_info or {})]


def parse_code():

    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT)

    reader = SimpleDirectoryReader(
        input_dir=SOURCE_CODE,
        file_extractor={".java": MyFileReader()},
        recursive=True,
        errors='backslashreplace',
        required_exts=[".java"],
        exclude=[".git", ".idea"]
    )

    documents = reader.load_data(num_workers=6, show_progress=True)

    logging.info("Read {}", len(documents))

    text_splitter = CodeSplitter.from_defaults(language='java')

    Settings.text_splitter = text_splitter
    # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    logging.info("Embedded model")

    vector_store = DuckDBVectorStore(database_name="my_vector_store.duckdb", persist_dir="duckdb_vectors")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        transformations=[text_splitter],
        storage_context=storage_context,
        show_progress=True)

    # save index to disk
    index.storage_context.persist()

    DuckDBVectorStore.persist(vector_store, persist_path="./duckdb_vectors")

    logging.info("Persisted data")

    # load index from disk

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)
    logging.info("Index creado")

    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=MODEL_URL,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 20},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    logging.info("Initialised llm")


    query_engine = index.as_query_engine(llm=llm)
    # Ask as many questions as you want against the loaded data:
    response = query_engine.query("Where is the most complex code in here?")
    logging.info(response)


def embedding_model():
    load_dotenv()

    FORMAT = '%(asctime)s %(user)-8s %(message)s'
    logging.basicConfig(format=FORMAT)

    model_name = "google/gemma-2b"

    token = os.getenv('HUGGINGFACE_TOKEN')

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=token)

    # Set pad token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load documents
    reader = SimpleDirectoryReader(
        input_dir=SOURCE_CODE,
        file_extractor={".java": MyFileReader()},
        recursive=True,
        errors='backslashreplace',
        required_exts=[".java"],
        exclude=[".git", ".idea"]
    )

    raw_documents = reader.load_data(num_workers=6, show_progress=True)

    # Split documents
    # text_splitter = CodeSplitter.from_defaults(language='java')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Create embeddings using the model and tokenizer
    inputs = tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()

    dimensions = embeddings.shape[1]

    logging.info("Found {} dimensions",dimensions)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parse_code()
    #embedding_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
