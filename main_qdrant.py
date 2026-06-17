from llama_index.core import SimpleDirectoryReader, Settings,Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core.indices.vector_store.base import VectorStoreIndex

Settings.embed_model =  HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")

SOURCE_CODE =  "/Users/josete/src/spring-ftdemo"

class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text + "java", extra_info=extra_info or {})]


def parse_code():

    client = qdrant_client.QdrantClient(url="http://192.168.1.43:6333", check_compatibility=False)

    vector_store = QdrantVectorStore(
        collection_name="ftdemo", client=client, 
    )

    reader = SimpleDirectoryReader(
        input_dir=SOURCE_CODE, file_extractor={".java": MyFileReader()}, recursive=True, errors='backslashreplace'
    )

    documents =  reader.load_data(num_workers=10)

    print("Read ",len(documents))

    llm = LMStudio(
        model_name="gemma-4-e4b-it",
        base_url="http://localhost:1234/v1",
        temperature=0.2,
    )


    vector_index = VectorStoreIndex.from_documents(documents, parallel=2,vector_store=vector_store, show_progress=True)
    vector_index.as_query_engine(llm=llm)

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

    print("Splitted text")

    # global


    Settings.text_splitter = text_splitter
    Settings.embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")

    print("Embedded model")


    # per-index
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store,transformations=[text_splitter])

    print(documents[1:2])


if __name__ == '__main__':
    parse_code()
