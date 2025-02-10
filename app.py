from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
import httpx
import re
import json
from typing import List, Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# API Keys & Model Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # For DeepSeek validation
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
DEEPSEEK_MODEL = "deepseek/deepseek-r1-distill-llama-70b"

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

logfire.configure(send_to_logfire="if-token-present")

# Define Pydantic models for the chat request
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# Create a dummy RunContext class to wrap our dependencies
class DummyRunContext:
    def __init__(self, deps: PydanticAIDeps):
        self.deps = deps

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are a Retrieval-Augmented Generation (RAG) agent specializing in legal research.
You retrieve and analyze information from a Supabase database containing processed PDF documents.

Your primary objectives:
- Provide highly accurate, legally precise responses.
- Quote the original text whenever possible, preserving its wording.
- Structure responses in full paragraphs, avoiding unnecessary numbering or bullet points.
- If an answer cannot be found, explicitly inform the user.
- ALWAYS cite the sources (document names, page and/or section, etc.) at the end of your response.
- Ensure that responses remain professional, well-structured, and suitable for legal professionals.

After generating your response, an additional AI (DeepSeek R1) will validate its accuracy.
"""

# Initialize the agent
model = OpenAIModel(LLM_MODEL)
pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

def extract_filters_from_query(query: str) -> Dict[str, str]:
    """
    Extract filtering criteria from the user query.
    E.g., if query contains "plaintiff", set party to "PLAINTIFF".
    Also, extract keywords for document title and a date if present.
    """
    q = query.lower()
    filters = {"party": "", "document_title": "", "date": ""}
    if "plaintiff" in q:
        filters["party"] = "PLAINTIFF"
    elif "defendant" in q:
        filters["party"] = "DEFENDANT"
    
    if "complaint" in q:
        filters["document_title"] = "complaint"
    elif "summons" in q:
        filters["document_title"] = "summons"
    elif "order" in q:
        filters["document_title"] = "order"
    
    date_match = re.search(r"\d{1,2}-[A-Za-z]{3}-\d{4}", query)
    if date_match:
        filters["date"] = date_match.group(0)
    return filters

@pydantic_ai_expert.tool
async def retrieve_relevant_pdf_chunks(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant PDF chunks based on the query using filtering.
    This function extracts filter criteria from the query (party, document_title, date)
    and queries the Supabase table accordingly, retrieving all columns.
    """
    print("🔍 Retrieving relevant PDF chunks with filtering...")
    filters = extract_filters_from_query(user_query)
    print("DEBUG: Filters extracted:", filters)
    
    try:
        query = ctx.deps.supabase.table("new_pdf_library").select("*")
        if filters.get("party"):
            query = query.eq("party", filters["party"])
        if filters.get("document_title"):
            query = query.ilike("document_title", f"%{filters['document_title']}%")
        if filters.get("date"):
            query = query.eq("date", filters["date"])
        
        result = query.execute()
        if not result.data or len(result.data) == 0:
            return "No relevant information was found in the database."
        
        response_parts = []
        source_documents = set()
        for doc in result.data:
            content = doc.get("content", "")
            document_name = doc.get("document_name", "Unknown Document")
            pages = doc.get("pages", [])
            citation = f"[Document: {document_name}, Pages: {', '.join(map(str, pages))}]"
            response_parts.append(f"{content}\n\n— {citation}")
            source_documents.add(citation)
        
        formatted_response = "\n\n".join(response_parts)
        source_info = "\n\n**Sources:** " + ", ".join(source_documents)
        return formatted_response + source_info

    except Exception as e:
        print(f"Error retrieving PDF documentation: {e}")
        return f"Error retrieving information: {str(e)}"

@pydantic_ai_expert.tool
async def generate_and_validate_response(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant chunks, generate an answer using GPT-4o, then validate with DeepSeek R1.
    """
    print("🔍 Step 1: Retrieving relevant PDF chunks...")
    retrieved_chunks = await retrieve_relevant_pdf_chunks(ctx, user_query)

    print("🚀 Step 2: Generating response using GPT-4o...")
    gpt_prompt = f"""
    You are a legal AI expert answering user questions based strictly on the provided legal text.

    **Rules for Your Response:**
    - Use **verbatim quotes** as much as possible.
    - Only paraphrase if necessary for coherence, and keep paraphrasing minimal.
    - Do not add any reasoning, interpretation, or conclusions beyond what is explicitly stated in the retrieved legal text.
    - Maintain a formal and professional tone.
    - Always cite the sources (document names, page/section numbers) as provided.

    ---
    **User Query:** {user_query}
    
    **Relevant Legal Chunks:** {retrieved_chunks}

    Now, generate a well-structured legal response following these rules.
    """
    gpt_response = await ctx.deps.openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": gpt_prompt}],
    )
    ai_response = gpt_response.choices[0].message.content

    print("✅ Step 3: Validating response with DeepSeek R1...")
    deepseek_validation = await validate_response_with_deepseek(ctx, user_query, retrieved_chunks, ai_response)

    print("✅ Process completed.")
    final_response = f"**AI Response:**\n{ai_response}\n\n**DeepSeek Validation:**\n{deepseek_validation}"
    return final_response

@pydantic_ai_expert.tool
async def validate_response_with_deepseek(
    ctx: RunContext[PydanticAIDeps],
    user_query: str,
    retrieved_chunks: str,
    ai_response: str
) -> str:
    """
    Validate the AI response using DeepSeek R1 to ensure accuracy and proper citation.
    """
    print("✅ Validating response with DeepSeek R1...")
    validation_prompt = f"""
    You are a legal AI validator responsible for ensuring AI-generated responses accurately reflect the legal text.

    **Your Task:**
    - Compare the AI response with the retrieved legal sources.
    - Ensure original wording is used as much as possible.
    - If there are errors, misinterpretations, or AI-inserted reasoning, highlight and correct them.
    - Confirm that citations (document names, page numbers, etc.) are accurate.

    ---
    **User Query:** {user_query}
    
    **Retrieved Legal Chunks:** {retrieved_chunks}
    
    **AI Response:** {ai_response}

    Now, generate a detailed validation report.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": validation_prompt}],
                    "temperature": 0.1
                }
            )
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error validating response with DeepSeek: {e}")
            return f"DeepSeek validation error: {str(e)}"

# FastAPI application to deploy on Hugging Face and connect to TypingMind.
from fastapi import FastAPI
app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """
    Endpoint to handle chat completions.
    The last message with role 'user' is used as the query.
    """
    user_query = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_query = msg.content
            break
    if not user_query:
        return {"error": "No user query found in messages."}
    
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )
    # Wrap dependencies in a dummy RunContext
    dummy_ctx = DummyRunContext(deps)
    
    result = await generate_and_validate_response(dummy_ctx, user_query)
    return {"choices": [{"message": {"role": "assistant", "content": result}}]}

# DummyRunContext class to wrap our dependencies
class DummyRunContext:
    def __init__(self, deps: PydanticAIDeps):
        self.deps = deps

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
