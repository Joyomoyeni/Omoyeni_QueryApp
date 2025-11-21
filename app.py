import os
import time
import requests
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables (like API key) from a .env file locally
load_dotenv()

# --- Configuration & Initialization ---
app = Flask(__name__)

# The API key must be secured via an environment variable on Render
# On Render, this is automatically injected. Locally, it comes from .env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # This check is crucial for security and deployment readiness
    print("FATAL ERROR: GEMINI_API_KEY environment variable not set.")
    # On deployment platforms, the app might still start but fail on API calls.

try:
    # Initialize the Gemini Client
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Use a faster model for chat/Q&A
    MODEL_NAME = "gemini-2.5-flash"
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None


# --- Utility for API Call with Exponential Backoff ---
def generate_content_with_retry(prompt, max_retries=5):
    """Handles the Gemini API call with exponential backoff for robustness."""
    if not client:
        return "ERROR: AI service is not initialized."

    retry_delay = 1  # Start with 1 second delay

    for attempt in range(max_retries):
        try:
            # Structure the content for the API call
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(prompt)])
            ]

            # Generate content
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
            return response.text.strip()

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(
                    f"API request failed (Attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s... Error: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Final API request failed after {max_retries} attempts. Error: {e}")
                return "AI Service Unavailable: Could not connect to the Generative AI model."
        except Exception as e:
            print(f"An unexpected error occurred during content generation: {e}")
            return "An internal server error occurred during processing."

    return "Failed to get a response after multiple attempts."


# --- API Endpoint ---
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    """Endpoint for receiving user queries and returning AI-generated answers."""
    data = request.get_json()
    user_query = data.get('query', '').strip()

    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    print(f"Received query: {user_query[:50]}...")  # Log the incoming query

    # Call the LLM to generate the response
    ai_response = generate_content_with_retry(user_query)

    if ai_response.startswith("ERROR:") or ai_response.startswith("AI Service Unavailable:"):
        # Handle errors returned by the utility function
        return jsonify({"error": ai_response}), 503

    return jsonify({
        "response": ai_response,
        "model": MODEL_NAME,
        "timestamp": time.time()
    })


# --- Simple Home Route for Health Check and Frontend Serving ---
@app.route('/')
def index():
    """Serves the main HTML page for the Q&A application."""
    # In a real setup, you would use 'render_template' or a static file handler.
    # For this demonstration, we'll return a simple health check message.
    return "LLM Q&A Backend is running. Access the application via the frontend UI."


if __name__ == '__main__':
    # When running locally, Flask uses the internal server.
    # On Render, Gunicorn (installed via requirements.txt) runs the app.
    # Use a dynamic port for local testing, though the host 0.0.0.0 is often needed for deployment platforms.
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))