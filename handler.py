# handler.py

import asyncio
import json
from typing import Dict, Any, List

from pydantic import BaseModel

# Import required items from your existing code.
# Ensure that app.py is in your repository root so that these imports work.
from app import (
    generate_and_validate_response,
    PydanticAIDeps,
    openai_client,
    supabase,
)

# DummyRunContext wraps our dependencies so that our tool functions can use ctx.deps
class DummyRunContext:
    def __init__(self, deps: PydanticAIDeps):
        self.deps = deps

# Define the EndpointHandler class as required by Hugging Face
class EndpointHandler:
    def __init__(self, model_path: str = ""):
        """
        Initialize the handler.
        In this example, our model logic is encapsulated in our generate_and_validate_response function,
        so we simply initialize our dependencies here.
        """
        from app import PydanticAIDeps, openai_client, supabase
        # Initialize dependencies from your app (ensure these objects are properly set in app.py)
        self.deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an incoming request.

        Parameters:
          data (Dict[str, Any]): The request body as a dictionary.
            It must include a key "messages" with a list of message objects.
            Each message object should have "role" and "content" fields.
        
        Returns:
          A list of dictionaries representing the output. This will be serialized to JSON.
        """
        # Ensure the input data is a dictionary
        if not isinstance(data, dict):
            return [{"error": "Input data must be a dictionary."}]
        
        # Extract the messages list from the payload.
        messages = data.get("messages", [])
        if not messages or not isinstance(messages, list):
            return [{"error": "Input must contain a 'messages' key with a list of messages."}]
        
        # Extract the last user message as the query.
        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
        
        if not user_query:
            return [{"error": "No user query found in messages."}]
        
        # Wrap our dependencies in a DummyRunContext
        dummy_ctx = DummyRunContext(self.deps)
        
        # Run the asynchronous generate_and_validate_response function synchronously.
        try:
            result = asyncio.run(generate_and_validate_response(dummy_ctx, user_query))
        except Exception as e:
            return [{"error": f"Inference error: {str(e)}"}]
        
        # Return the result in a list (as Hugging Face expects a list of outputs).
        return [{"message": {"role": "assistant", "content": result}}]

# For local testing of the handler (optional)
if __name__ == "__main__":
    # Define a sample payload matching the expected format.
    test_payload = {
        "messages": [
            {
                "role": "user",
                "content": "What does the defendant state in paragraph 15 of their Answer?"
            }
        ]
    }
    # Create an instance of the handler.
    handler = EndpointHandler()
    # Process the test payload.
    output = handler(test_payload)
    # Print the output as formatted JSON.
    print(json.dumps(output, indent=2))
