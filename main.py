# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

SOURCE_CODE = "/home/josete/src/spring-fault-tolerance"


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class MyFileReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r") as f:
            text = f.read()
        # load_data returns a list of Document objects
        return [Document(text=text, extra_info=extra_info or {})]


def parse_code():
    reader = SimpleDirectoryReader(
        input_dir=SOURCE_CODE, file_extractor={".java": MyFileReader()}, recursive=True, errors='backslashreplace'
    )

    documents = reader.load_data(num_workers=4)

    print("Read ", len(documents))

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

    Settings.text_splitter = text_splitter
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Embedded model")

    # per-index
    index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter], show_progress=True)

    index.as_query_engine()

    print("Vector index creado")

    print(documents[1:2])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    parse_code()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
