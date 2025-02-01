import chromadb


class ChromaProcessing:
    """
    A class to handle operations related to Chroma knowledge bases.

    Methods
    -------
    __init__():
        Initializes the ChromaProcessing class with a persistent Chroma client.

    delete_knowledge_base(doc_id: str):
        Deletes a knowledge base collection by its document ID.

    add_new_knowledge_base(name: str):
        Creates a new knowledge base collection with the specified name and metadata.

    add_document_to_knowledge_base(kb: str, content: list):
        Adds a list of documents to the specified knowledge base collection.

    list_knowledge_base() -> list:
        Lists all the knowledge base collections.

    retrieve_chunks(kb_id: str, content: str) -> list:
        Retrieves chunks of data from the specified knowledge base collection based on the query content.
    """
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma/")

    def delete_knowledge_base(self, doc_id: str):
        self.chroma_client.delete_collection(name=doc_id)

    def add_new_knowledge_base(self, name: str):
        self.chroma_client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine", "hnsw:num_threads": 10}
        )

    def add_document_to_knowledge_base(self, kb: str, content: list):
        self.chroma_client.get_collection(kb).add(
            documents=content, ids=[str(i) for i in range(0, len(content))]
        )

    def list_knowledge_base(self) -> list:
        return self.chroma_client.list_collections()

    def retrieve_chunks(self, kb_id: str, content: str) -> list:
        knowledge_base = self.chroma_client.get_collection(name=kb_id)

        results = knowledge_base.query(query_texts=[content], n_results=2)

        return results
