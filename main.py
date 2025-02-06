import asyncio
import json
import os
import websockets
import google.generativeai as genai
import base64

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# Load API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash-exp"

if not gemini_api_key:
    raise ValueError("API key must be set. Export GEMINI_API_KEY in your environment.")

# Configure the API before initializing the client
genai.configure(api_key=gemini_api_key)

# Initialize the Gemini client
client = genai.GenerativeModel(model_name=MODEL)

# Initialize embedding and LLM models
gemini_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/text-embedding-004")
llm = Gemini(api_key=gemini_api_key, model_name="models/gemini-2.0-flash-exp")

def build_index(doc_path="./downloads"):
    # Check if storage already exists
    Settings.llm = llm
    Settings.embed_model = gemini_embedding_model
    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        # Load the documents and create the index
        documents = SimpleDirectoryReader(doc_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        # Store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

def query_docs(query):
    index = build_index()
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    # Convert the response to a string
    response_text = str(response)
    print(f"RAG response: {response_text}")
    return response_text

# Define the tool (function)
tool_query_docs = {
    "function_declarations": [
        {
            "name": "query_docs",
            "description": "Query the document content with a specific query string.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The query string to search the document index."
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

async def send_to_gemini(client_websocket):
    """Sends messages from the client websocket to the Gemini API."""
    try:
        async for message in client_websocket:
            try:
                data = json.loads(message)
                if "realtime_input" in data:
                    for chunk in data["realtime_input"]["media_chunks"]:
                        if chunk["mime_type"] == "audio/pcm":
                            # Handle audio data (if needed)
                            pass
                        elif chunk["mime_type"] == "application/pdf":
                            # Save PDF file to downloads directory
                            pdf_data = base64.b64decode(chunk["data"])
                            filename = chunk.get("filename", "uploaded.pdf")
                            
                            # Create downloads directory if it doesn't exist
                            os.makedirs("./downloads", exist_ok=True)
                            
                            # Save the PDF file
                            file_path = os.path.join("./downloads", filename)
                            with open(file_path, "wb") as f:
                                f.write(pdf_data)
                            
                            print(f"Saved PDF file to {file_path}")
                            
                            # Rebuild the index with the new PDF
                            if os.path.exists("./storage"):
                                import shutil
                                shutil.rmtree("./storage")
                            build_index()
                            
                            await client_websocket.send(json.dumps({
                                "text": f"PDF file {filename} has been uploaded and indexed successfully."
                            }))
                                
            except Exception as e:
                print(f"Error sending to Gemini: {e}")
        print("Client connection closed (send)")
    except Exception as e:
        print(f"Error sending to Gemini: {e}")
    finally:
        print("send_to_gemini closed")

async def receive_from_gemini(session, client_websocket):
    """Receives responses from the Gemini API and forwards them to the client."""
    try:
        while True:
            try:
                print("receiving from gemini")
                # Send a query to Gemini and get the response
                response = session.send_message("Your query here")  # Replace with actual query

                # Check if the response contains text
                if response.text:
                    await client_websocket.send(json.dumps({"text": response.text}))

                # Check if the response contains tool calls (if applicable)
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call'):
                                    # Handle tool calls
                                    function_call = part.function_call
                                    function_name = function_call.name
                                    function_args = function_call.args

                                    if function_name == "query_docs":
                                        try:
                                            result = query_docs(function_args["query"])
                                            function_response = {
                                                "name": function_name,
                                                "response": {"result": result}
                                            }
                                            await client_websocket.send(json.dumps({"text": json.dumps(function_response)}))
                                            print("Function executed")
                                        except Exception as e:
                                            print(f"Error executing function: {e}")
                                            continue

                # Check if the turn is complete
                if hasattr(response, 'turn_complete') and response.turn_complete:
                    print('\n<Turn complete>')
                    break

            except websockets.exceptions.ConnectionClosedOK:
                print("Client connection closed normally (receive)")
                break
            except Exception as e:
                print(f"Error receiving from Gemini: {e}")
                break

    except Exception as e:
        print(f"Error receiving from Gemini: {e}")
    finally:
        print("Gemini connection closed (receive)")

async def gemini_session_handler(client_websocket: websockets.WebSocketServerProtocol):
    """Handles the interaction with Gemini API within a websocket session."""
    try:
        config_message = await client_websocket.recv()
        config_data = json.loads(config_message)
        config = config_data.get("setup", {})
        config["system_instruction"] = """You are a helpful assistant and you MUST always use the query_docs tool to query the document 
        towards any questions. It is mandatory to base your answers on the information from the output of the query_docs tool, 
        and include the context from the query tool in your response to the user's question.
        Do not mention your operations like "I am searching the document now".
        """

        config["tools"] = [tool_query_docs]

        # Start a synchronous Gemini chat session
        session = client.start_chat(history=[])
        print("Connected to Gemini API")

        # Start send and receive tasks
        send_task = asyncio.create_task(send_to_gemini(client_websocket))
        receive_task = asyncio.create_task(receive_from_gemini(session, client_websocket))
        await asyncio.gather(send_task, receive_task)

    except Exception as e:
        print(f"Error in Gemini session: {e}")
    finally:
        print("Gemini session closed.")

async def main() -> None:
    async with websockets.serve(gemini_session_handler, "localhost", 9084):
        print("Running websocket server localhost:9084...")
        await asyncio.Future()  # Keep the server running indefinitely

if __name__ == "__main__":
    asyncio.run(main())