# main.py
import asyncio
import json
import os
import websockets
from google import genai
import base64
from tenacity import retry, wait_exponential, stop_after_attempt
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

class GeminiChatServer:
    def __init__(self):
        # Load API key from environment
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY', '')
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
            
        self.MODEL = "gemini-2.0-flash-exp"
        self.text_embedding_model = "text-embedding-004"
        
        # Initialize Gemini client
        self.client = genai.Client(
            http_options={'api_version': 'v1alpha'}
        )
        
        # Initialize models
        self.gemini_embedding_model = GeminiEmbedding(
            api_key=self.gemini_api_key, 
            model_name="models/text-embedding-004"
        )
        
        self.llm = Gemini(
            api_key=self.gemini_api_key, 
            model_name="models/gemini-2.0-flash-exp"
        )
        
        # Initialize settings
        Settings.llm = self.llm
        Settings.embed_model = self.gemini_embedding_model
        
        # Create necessary directories
        Path("./downloads").mkdir(exist_ok=True)
        Path("./storage").mkdir(exist_ok=True)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def build_index(self, doc_path="./downloads"):
        """Build or load document index with retry logic"""
        PERSIST_DIR = "./storage"
        
        try:
            if not os.path.exists(PERSIST_DIR):
                documents = SimpleDirectoryReader(doc_path).load_data()
                index = VectorStoreIndex.from_documents(documents)
                index.storage_context.persist(persist_dir=PERSIST_DIR)
            else:
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            print(f"Error building index: {e}")
            raise

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3))
    def query_docs(self, query):
        """Query documents with retry logic"""
        try:
            index = self.build_index()
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            response_text = str(response)
            print(f"RAG response: {response_text}")
            return response_text
        except Exception as e:
            print(f"Error querying docs: {e}")
            raise

    async def handle_session(self, websocket: websockets.WebSocketServerProtocol):
        """Handle websocket session"""
        try:
            config_message = await websocket.recv()
            config_data = json.loads(config_message)
            config = config_data.get("setup", {})
            
            # Set system instruction and tools
            config["system_instruction"] = """You are a helpful assistant and you MUST always use the query_docs tool to query the document 
            towards any questions. Base your answers on the tool's output and include relevant context in your responses.
            Maintain a natural conversational tone."""
            
            config["tools"] = [{
                "function_declarations": [{
                    "name": "query_docs",
                    "description": "Query the document content",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {
                                "type": "STRING",
                                "description": "Query string to search documents"
                            }
                        },
                        "required": ["query"]
                    }
                }]
            }]

            async with self.client.aio.live.connect(model=self.MODEL, config=config) as session:
                print("Connected to Gemini API")
                await self.manage_session(websocket, session)
                
        except Exception as e:
            print(f"Session error: {e}")
            await websocket.close(1011, f"Internal error: {str(e)}")

    async def manage_session(self, websocket, session):
        """Manage the active session"""
        send_task = asyncio.create_task(self.send_to_gemini(websocket, session))
        receive_task = asyncio.create_task(self.receive_from_gemini(websocket, session))
        
        try:
            await asyncio.gather(send_task, receive_task)
        except Exception as e:
            print(f"Session management error: {e}")
        finally:
            for task in [send_task, receive_task]:
                if not task.done():
                    task.cancel()

    async def send_to_gemini(self, websocket, session):
        """Handle sending messages to Gemini"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if "realtime_input" in data:
                        await self.process_input(data, websocket, session)
                except Exception as e:
                    print(f"Error processing message: {e}")
        except Exception as e:
            print(f"Send to Gemini error: {e}")

    async def process_input(self, data, websocket, session):
        """Process different types of input"""
        for chunk in data["realtime_input"]["media_chunks"]:
            if chunk["mime_type"] == "audio/pcm":
                await session.send(input={
                    "mime_type": "audio/pcm",
                    "data": chunk["data"]
                })
            elif chunk["mime_type"] == "application/pdf":
                await self.handle_pdf_upload(chunk, websocket)

    async def handle_pdf_upload(self, chunk, websocket):
        """Handle PDF file upload"""
        try:
            pdf_data = base64.b64decode(chunk["data"])
            filename = chunk.get("filename", "uploaded.pdf")
            file_path = os.path.join("./downloads", filename)
            
            with open(file_path, "wb") as f:
                f.write(pdf_data)
            
            print(f"Saved PDF: {file_path}")
            
            if os.path.exists("./storage"):
                import shutil
                shutil.rmtree("./storage")
            
            self.build_index()
            
            await websocket.send(json.dumps({
                "text": f"PDF file {filename} uploaded and indexed successfully."
            }))
        except Exception as e:
            print(f"PDF handling error: {e}")
            await websocket.send(json.dumps({
                "text": f"Error processing PDF: {str(e)}"
            }))

    async def receive_from_gemini(self, websocket, session):
        """Handle receiving messages from Gemini"""
        try:
            while True:
                async for response in session.receive():
                    await self.process_gemini_response(response, websocket, session)
        except Exception as e:
            print(f"Receive from Gemini error: {e}")

    async def process_gemini_response(self, response, websocket, session):
        """Process Gemini API responses"""
        try:
            if response.server_content is None:
                if response.tool_call is not None:
                    await self.handle_tool_call(response.tool_call, websocket, session)
                return

            model_turn = response.server_content.model_turn
            if model_turn:
                await self.handle_model_turn(model_turn, websocket)

        except Exception as e:
            print(f"Response processing error: {e}")

    async def handle_tool_call(self, tool_call, websocket, session):
        """Handle tool calls from Gemini"""
        function_responses = []
        
        for function_call in tool_call.function_calls:
            try:
                if function_call.name == "query_docs":
                    result = self.query_docs(function_call.args["query"])
                    function_responses.append({
                        "name": function_call.name,
                        "response": {"result": result},
                        "id": function_call.id
                    })
            except Exception as e:
                print(f"Tool call error: {e}")
                continue
        
        if function_responses:
            await websocket.send(json.dumps({"text": json.dumps(function_responses)}))
            await session.send(input=function_responses)

    async def handle_model_turn(self, model_turn, websocket):
        """Handle model turn responses"""
        for part in model_turn.parts:
            if hasattr(part, 'text') and part.text:
                await websocket.send(json.dumps({"text": part.text}))
            elif hasattr(part, 'inline_data') and part.inline_data:
                base64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                await websocket.send(json.dumps({"audio": base64_audio}))

async def main():
    server = GeminiChatServer()
    async with websockets.serve(server.handle_session, "localhost", 9084):
        print("Server running on ws://localhost:9084")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())