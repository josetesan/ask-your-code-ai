from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings,Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

SOURCE_CODE =  "/Users/ou83mp/Developer/src/techday/spring-ftdemo"

class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text + "java", extra_info=extra_info or {})]


def parse_code():

    reader = SimpleDirectoryReader(
        input_dir=SOURCE_CODE, file_extractor={".java": MyFileReader()}, recursive=True, errors='backslashreplace'
    )

    documents =  reader.load_data(num_workers=10)

    print("Read ",len(documents))

    llm = Ollama(model="llama3.1", request_timeout=600.0)

    vector_index = VectorStoreIndex.from_documents(documents,show_progress=True)
    vector_index.as_query_engine(llm=llm)


    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

    print("Splitted text")

    # global


    Settings.text_splitter = text_splitter
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Embedded model")


    # per-index
    index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])

    print(documents[1:2])


if __name__ == '__main__':
    parse_code()
