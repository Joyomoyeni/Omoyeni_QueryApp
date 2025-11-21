# LLM_QA_CLI.py
import os
import re
from google import genai
from google.genai.errors import APIError


def preprocess_question(question: str) -> str:
    """Applies basic preprocessing: lowercasing, tokenization, punctuation removal."""
    # 1. Lowercasing
    processed_q = question.lower()
    # 2. Punctuation removal (keeping spaces)
    processed_q = re.sub(r'[^\w\s]', '', processed_q)
    # 3. Simple tokenization (split by space) - implicitly done by Python/LLM later,
    # but cleaning up multiple spaces is good practice.
    processed_q = re.sub(r'\s+', ' ', processed_q).strip()
    return processed_q


def get_llm_answer(question: str, preprocessed_q: str) -> str:
    """Constructs prompt, calls Gemini API, and returns the answer."""
    try:
        # Initialize the client
        # The key will be read from the GEMINI_API_KEY environment variable
        client = genai.Client()

        # Construct a clear prompt for the LLM
        prompt = (
            f"Based on the original question: '{question}', provide a concise and helpful answer."
            f"\n\n[Internal/Preprocessed Query]: {preprocessed_q}"
        )

        # Call the model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        return response.text

    except APIError as e:
        return f"An LLM API Error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def main():
    """Main function to run the CLI application."""
    print("🤖 LLM Question-and-Answering CLI System")

    # Get question from user input
    user_question = input("Ask a question: ")

    if not user_question:
        print("Please enter a question.")
        return

    # 1. Preprocessing
    print("\n--- Processing ---")
    processed_question = preprocess_question(user_question)
    print(f"Original Question: {user_question}")
    print(f"Processed Query:   {processed_question}")

    # 2. Get Answer from LLM
    print("Asking the LLM... Please wait.")
    llm_answer = get_llm_answer(user_question, processed_question)

    # 3. Display Result
    print("\n--- Final Answer ---")
    print(llm_answer)
    print("--------------------")


if __name__ == "__main__":
    main()