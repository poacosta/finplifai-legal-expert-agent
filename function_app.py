import json
import logging
import os
import azure.functions as func

import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent import ReActAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-function")

# Get configuration from environment variables
CHROMA_HOST = os.environ.get("CHROMA_HOST")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "knowledge")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "o3-mini")
OPENAI_EMBEDDING_MODEL_NAME = os.environ.get(
    "OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small"
)
FUNCTION_API_TOKEN = os.environ.get("FUNCTION_API_TOKEN")
VERBOSE = os.environ.get("VERBOSE", "false").lower() == "true"

# Initialize agent outside the handler to improve cold start performance
agent = None

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


def initialize_agent():
    """Initialize the LLM agent with ChromaDB vector store."""
    global agent

    if agent is not None:
        return

    # Validate required environment variables
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set")

    if not FUNCTION_API_TOKEN:
        logger.error("FUNCTION_API_TOKEN environment variable not set")
        raise ValueError("FUNCTION_API_TOKEN environment variable not set")

    if not CHROMA_HOST:
        logger.error("CHROMA_HOST environment variable not set")
        raise ValueError("CHROMA_HOST environment variable not set")

    # Set OpenAI API key in environment
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    logger.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
    try:
        # Initialize ChromaDB client
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        chroma_collection = client.get_collection(CHROMA_COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Initialize embeddings and LLM
        embed_model = OpenAIEmbedding(
            model=OPENAI_EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY
        )

        llm = OpenAI(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

        # Configure global settings
        Settings.embed_model = embed_model
        Settings.llm = llm

        # Create index and query engine
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

        query_engine = index.as_query_engine(llm=llm)

        # Create query engine tool
        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="legal_expert_accountant",
            description="an expert accountant on the economic laws and regulations of Spain",
            return_direct=False,
        )

        # Initialize agent
        agent = ReActAgent.from_tools(
            tools=[query_engine_tool], llm=llm, verbose=VERBOSE
        )
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


def run_query(query_text):
    """Execute a query against the agent.

    Args:
        query_text: The query string to process

    Returns:
        Dict containing the response and metadata
    """
    try:
        logger.info(f"Processing query: {query_text}")
        response = agent.query(query_text)

        # Handle different response types from the agent
        if hasattr(response, "response"):
            # For newer versions of LlamaIndex that return a response object
            result = response.response
        else:
            # For versions that return a string directly
            result = str(response)

        return {"result": result, "status": "success"}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e), "status": "error"}


@app.route(route="query")
def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function HTTP trigger.

    Args:
        req: The HTTP request

    Returns:
        HTTP response
    """
    logger.info("Python HTTP trigger function processed a request")

    # Initialize agent if not already done
    try:
        initialize_agent()
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        return func.HttpResponse(
            body=json.dumps(
                {"status": "error", "message": "Failed to initialize agent"}
            ),
            mimetype="application/json",
            status_code=500,
        )

    # Handle CORS preflight requests
    if req.method == "OPTIONS":
        return func.HttpResponse(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            },
        )

    # Only allow POST method
    if req.method != "POST":
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": "Method not allowed"}),
            mimetype="application/json",
            status_code=405,
        )

    # Validate authentication token
    auth_header = req.headers.get("Authorization")

    if not auth_header:
        logger.warning("Missing Authorization header")
        return func.HttpResponse(
            body=json.dumps(
                {"status": "error", "message": "Missing authentication token"}
            ),
            mimetype="application/json",
            status_code=401,
        )

    # Extract token (support for both "Bearer token" and "token" formats)
    token_parts = auth_header.split()
    received_token = token_parts[-1] if len(token_parts) > 1 else token_parts[0]

    if received_token != FUNCTION_API_TOKEN:
        logger.warning("Invalid authentication token")
        return func.HttpResponse(
            body=json.dumps(
                {"status": "error", "message": "Invalid authentication token"}
            ),
            mimetype="application/json",
            status_code=403,
        )

    # Parse request body
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            body=json.dumps(
                {"status": "error", "message": "Invalid JSON in request body"}
            ),
            mimetype="application/json",
            status_code=400,
        )

    # Get query from request body
    query = req_body.get("query")
    if not query:
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": "Missing query parameter"}),
            mimetype="application/json",
            status_code=400,
        )

    # Run the query
    try:
        result = run_query(query)
        status_code = 200 if result.get("status") == "success" else 500

        return func.HttpResponse(
            body=json.dumps(result),
            mimetype="application/json",
            status_code=status_code,
            headers={"Access-Control-Allow-Origin": "*"},
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": "Internal server error"}),
            mimetype="application/json",
            status_code=500,
        )
