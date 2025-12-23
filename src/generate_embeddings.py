import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

def get_module_code(module_path):
    """Reads the content of a Python module file."""
    with open(module_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_code_embeddings(code_snippets: dict):
    """Generates embeddings for code snippets using Google's Generative AI.
    
    Args:
        code_snippets (dict): A dictionary where keys are module names and values are code strings.
    
    Returns:
        dict: A dictionary where keys are module names and values are their embeddings.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set the environment variable.")
    
    genai.configure(api_key=api_key)
    
    # For now, this is a placeholder. In a real scenario, you'd use a model that supports embeddings.
    # For example: model = genai.GenerativeModel('embedding-001')
    # embeddings = {name: model.embed_content(content=code) for name, code in code_snippets.items()}
    
    print("\n--- Placeholder for Code Embedding Generation ---")
    print("Embeddings would be generated here using a Google Generative AI embedding model.")
    print("For demonstration, returning mock embeddings.")
    
    mock_embeddings = {}
    for name in code_snippets.keys():
        # Mock embedding: a list of zeros for demonstration
        mock_embeddings[name] = [0.0] * 768  # Common embedding size, adjust as needed
        
    return mock_embeddings


if __name__ == "__main__":
    src_directory = "src"
    python_files = [
        f for f in os.listdir(src_directory) 
        if f.endswith('.py') and f != '__init__.py' and f != 'chatbot.py' and f != 'main.py'
    ]

    code_snippets = {}
    for filename in python_files:
        module_name = filename.replace('.py', '')
        file_path = os.path.join(src_directory, filename)
        code_snippets[module_name] = get_module_code(file_path)
        print(f"Loaded code for {module_name}.py")

    if code_snippets:
        try:
            embeddings = generate_code_embeddings(code_snippets)
            print("\nGenerated Embeddings (sample for one module):")
            # Print a sample of the embeddings
            for module, embed_vector in list(embeddings.items())[:1]:
                print(f"  {module}: {embed_vector[:5]}... (first 5 elements)")
            print("\nCode embeddings generation process completed (with mock embeddings).")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("No Python files found in src/ to generate embeddings for.")
