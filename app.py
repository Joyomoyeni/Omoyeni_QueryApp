# app.py
import os
from flask import Flask, render_template, request
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv  # Use this for local testing

# Load environment variables (API Key) from a .env file locally
load_dotenv()

app = Flask(__name__)


# Re-use the preprocessing logic from PART A (or adapt it)
def preprocess_question(question: str) -> str:
    """Applies basic preprocessing."""
    # Simplified for the web example, but full logic from CLI should be here
    return question.lower().strip()


def get_llm_answer_web(question: str, preprocessed_q: str) -> str:
    """Constructs prompt, calls Gemini API, and returns the answer."""
    try:
        # Check if API Key is available
        if 'GEMINI_API_KEY' not in os.environ:
            return "API Key not found. Please set the GEMINI_API_KEY environment variable."

        client = genai.Client()
        prompt = (
            f"Based on the original question: '{question}', provide a concise and helpful answer."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        return response.text

    except APIError as e:
        return f"An LLM API Error occurred: {e}. Check your key and permissions."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


@app.route('/', methods=['GET', 'POST'])
def index():
    user_question = None
    processed_query = None
    llm_answer = None

    if request.method == 'POST':
        user_question = request.form['question']

        if user_question:
            # 1. View the processed question
            processed_query = preprocess_question(user_question)

            # 2. Get LLM API response/generated answer
            llm_answer = get_llm_answer_web(user_question, processed_query)

    # 3. Display the generated answer, original, and processed query
    return render_template(
        'index.html',
        question=user_question,
        processed=processed_query,
        answer=llm_answer
    )


if __name__ == '__main__':
    # Ensure the API key is set for deployment
    if 'GEMINI_API_KEY' not in os.environ:
        print("WARNING: GEMINI_API_KEY environment variable not set. The app will fail during API call.")

    app.run(debug=True)