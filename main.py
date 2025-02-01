from fastapi import FastAPI, UploadFile
from pypdf import PdfReader
from textwrap import wrap
import google.generativeai as genai
from gensim.parsing.preprocessing import remove_stopwords
from chroma_processing import ChromaProcessing

app = FastAPI()
chroma_client = ChromaProcessing()


@app.get("/status")
async def status():
    """
    Asynchronous function to return the status of the application.

    Returns:
        dict: A dictionary containing the status message.
    """
    return {"status": "it works"}


@app.get("/knowledge_bases/")
async def list_knowledge_bases():
    """
    Asynchronously lists all knowledge bases.

    This function interacts with the `chroma_client` to retrieve a list of all
    knowledge bases. If an error occurs during the process, it catches the 
    exception and returns an error message.

    Returns:
        dict: A dictionary containing either the list of knowledge bases or an 
        error message if an exception occurs.
    """
    try:
        response = chroma_client.list_knowledge_base()
        return {"list of knowledge bases": response}

    except Exception as e:
        return {"message": f"error occured {e}"}


@app.delete("/knowledge_base/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """
    Asynchronously deletes a knowledge base by its ID.

    Args:
        kb_id (str): The ID of the knowledge base to be deleted.

    Returns:
        dict: A dictionary containing a success message if the deletion is successful,
              or an error message if an exception occurs.
    """
    try:
        chroma_client.delete_knowledge_base(kb_id)
        return {"message": f"{kb_id} deleted successfully"}
    except Exception as e:
        return {"message": f"error occurred {e}"}


@app.post("/create_knowledge_base/{kb_id}")
async def create_knowledge_base(kb_id: str):
    """
    Asynchronously creates a new knowledge base with the given ID.

    Args:
        kb_id (str): The ID of the knowledge base to be created.

    Returns:
        dict: A dictionary containing a success message if the knowledge base is created successfully,
              or an error message if an exception occurs.
    """
    try:
        chroma_client.add_new_knowledge_base(name=kb_id)
        return {"message": "new knowledge base is successfully created"}
    except Exception as e:
        return {"message": "error occured {e}"}


@app.post("/add_document_to_knowledge_base/{kb_id}")
async def upload_file(kb_id: str, file: UploadFile):
    """
    Uploads a PDF file, extracts its text, processes it, and stores it in a knowledge base.

    Args:
        kb_id (str): The ID of the knowledge base where the document will be stored.
        file (UploadFile): The PDF file to be uploaded.

    Returns:
        dict: A message indicating the success or failure of the operation.
            - If successful: {"message": "pdf is successfully chunked and stored into {kb_id}"}
            - If an error occurs: {"message": "something went wrong : {e}"}
            - If the file is not a PDF: {"message": "we support only pdfs"}
    """
    if file.content_type == "application/pdf":
        try:
            text = ""
            reader = PdfReader(file.file)

            for page in reader.pages:
                text += page.extract_text()

            text = text.strip()
            text = remove_stopwords(text)

            chunks = wrap(text, 1024)

            chroma_client.add_document_to_knowledge_base(kb=kb_id, content=chunks)

            return {"message": f"pdf is successfully chunked and stored into {kb_id}"}

        except Exception as e:
            return {"message": f"something went wrong : {e}"}
    else:
        return {"message": "we support only pdfs"}


@app.post("/query/{kb_id}")
async def query(kb_id: str, query: str):
    """
    Asynchronously queries a knowledge base and generates a formal response using a generative AI model.

    Args:
        kb_id (str): The ID of the knowledge base to query.
        query (str): The query string to process and analyze.

    Returns:
        dict: A dictionary containing the generated response text under the key "result".
              If an error occurs, a dictionary with an error message under the key "message".

    Raises:
        Exception: If there is an issue retrieving data from the knowledge base.
    """

    try:
        query_without_stop_words = remove_stopwords(query)
        results = chroma_client.retrieve_chunks(kb_id= kb_id, content=query_without_stop_words)

        genai.configure(api_key="Your API Key")
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=f"""
            You are an AI assistant chatbot, You will use this text to response :
            provided text = {results}
            Read & Analyse this text then create a formal response using only the provided text.
            If information doesn't exist in the provided text, respond by the following sentence : It does not exist.
            Your response must be in English.
            """,
        )
        response = model.generate_content(query)

        return {"result": response.text}

    except Exception as e:
        return {"message": f"I couldn't retrieve data from chromadb {e}"}
