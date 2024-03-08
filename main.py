from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
    load_index_from_storage
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import (
    CodeSplitter
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.vector_stores.duckdb import DuckDBVectorStore
from IPython.display import Markdown, display

SOURCE_CODE = '/home/josete/src/spring-fault-tolerance'
# MODEL_URL = "https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q8_0.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q2_K.gguf"


class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text, extra_info=extra_info or {})]


def parse_code():
    reader = SimpleDirectoryReader(
        input_dir=SOURCE_CODE,
        file_extractor={".java": MyFileReader()},
        recursive=True,
        errors='backslashreplace',
        required_exts=[".java"],
        exclude=[".git", ".idea"]
    )

    documents = reader.load_data(num_workers=6, show_progress=True)

    print("Read ", len(documents))

    text_splitter = CodeSplitter.from_defaults(language='java')

    Settings.text_splitter = text_splitter
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Embedded model")

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

    # load index from disk

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context=storage_context)
    print("Index creado")

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

    print("Initialised llm")

    query_engine = index.as_query_engine(llm=llm)
    # Ask as many questions as you want against the loaded data:
    response = query_engine.query("Where is the most complex code in here?")
    print(response)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parse_code()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
