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
    CodeSplitter,
    MarkdownNodeParser
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.vector_stores.duckdb import DuckDBVectorStore
import gradio as gr
import traceback
import logging

#SOURCE_CODE = '/home/josete/src/spring-fault-tolerance'
SOURCE_CODE = '/Users/ou83mp/Developer/src/EngineeringProductivity/P16575-engineering-journey/src/pages'
# MODEL_URL = "https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q8_0.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q2_K.gguf"

class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text, extra_info=extra_info or {})]


def initialise():

    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT)


    reader = SimpleDirectoryReader(
        input_dir=SOURCE_CODE,
        file_extractor={".md": MyFileReader()},
        recursive=True,
        errors='backslashreplace',
        required_exts=[".md"],
        exclude=[".git", ".idea",".png",".gif",".svg",".astro"]
    )

    documents = reader.load_data(num_workers=6, show_progress=True)

    print("Read ", len(documents))

    text_splitter = MarkdownNodeParser()

    Settings.text_splitter = text_splitter
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Embedded model")

    vector_store = DuckDBVectorStore(database_name="my_vector_store.duckdb", persist_dir="duckdb_vectors")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    try:
        index = VectorStoreIndex.from_documents(
            documents=documents,
            transformations=[text_splitter],
            storage_context=storage_context,
            show_progress=True)
    except Exception as e:
        traceback.print_exc(e)
        raise e

    # save index to disk
    try:
       index.storage_context.persist()
    except Exception as e:
        traceback.print_exc(e)
        raise e
    
    try:
        DuckDBVectorStore.persist(vector_store, persist_path="./duckdb_vectors")
    except Exception as e:
        traceback.print_exc(e)
        raise e

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
        model_kwargs={"n_gpu_layers": 0},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    print("Initialised llm")

    query_engine = index.as_query_engine(llm=llm)

    return query_engine


def ask_the_journey(input_text):
    response = query_engine.query(input_text)
    print(response)
    return response 


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    query_engine = initialise()
    prompt = """
He ejecutado el pipeline para inicializar DBaaS y para crear la matriz de SollAR y las NPAs que se han creado han sido SCHEMANAME_OWNER y USER con sus proxies (PROXY1, PROXY2) y los READ. Sin embargo, no hay forma de saber si son para TST, ACC o PRD (Tenemos otro asset en que las NPAs son SCHEMANAME_ENV_OWNER, etc.)
 
Esto está bien? Es posible meter la misma NPA en diferentes safes (tenemos 3, TST, ACC y PRD)?"""
    response = query_engine.query(prompt)
    print(response)

#    iface = gr.Interface(
#        fn=ask_the_journey,
#        inputs=gr.Textbox(lines=3, placeholder="Enter your query here"),
#        outputs=gr.Markdown(),
#        title="Journey chatbot",
#        description="EP Journey chatbot"
#    )

#    iface.launch()

