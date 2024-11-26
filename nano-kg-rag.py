"""
Requirements:

In case we run it in https://colab.research.google.com/

# Install requirements
!apt-get update
!apt-get install -y sudo
!pip install rdflib llama-index-llms-ollama

# Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh
!ollama --version

# Start Ollama in the background
!nohup ollama start > ollama_log.out 2>&1 &

# Launch Ollama endpoint
!ollama run llama3.1:8b > /dev/null 2>&1 &

# To run it locally
Install Ollama locally https://ollama.com/download
ollama run llama3.1:8b
python3 -m venv venv && source venv/bin/activate
python3 -m pip install llama-index-llms-ollama rdflib

# ---

User's questions we are going to cover:

Donna is Nathan's sister - please remember this.
Katie is Nathan's wife - please remember this.
What is the name of Nathan's spouse?
Please remember: Johnny is Katie's brother.
What is the name of Katie's brother?
What is the name of Johnny's sister?
Katie has brother Johnny.
Who is Katie's brother?

"""

# ---

import rdflib
from llama_index.llms.ollama import Ollama

# Constants
KNOW_ONTOLOGY_URL = "https://know.dev/"
MODEL_NAME = 'llama3.1:8b'
TURTLE_FILE = "knowledge_base.ttl"

QUERY_ONTOLOGY_SUBSET = '''
    PREFIX know: <https://know.dev/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    CONSTRUCT {
    ?s ?p ?o .
    }
    WHERE {
    {
        # Include all triples where the subject is the Person class
        know:Person ?p ?o .
        BIND(know:Person AS ?s)
        # Filter out non-English labels
        FILTER (!isLiteral(?o) || lang(?o) = "" || lang(?o) = "en")
    }
    UNION
    {
        # Include all properties where the domain is Person
        ?s rdfs:domain know:Person ;
        ?p ?o .
        # Filter out non-English labels
        FILTER (!isLiteral(?o) || lang(?o) = "" || lang(?o) = "en")
    }
    UNION
    {
        # Include all properties where the range is Person
        ?s rdfs:range know:Person ;
        ?p ?o .
        # Filter out non-English labels
        FILTER (!isLiteral(?o) || lang(?o) = "" || lang(?o) = "en")
    }
    }
'''

# Load the RDF ontology file
def load_ontology(url):
    graph = rdflib.Graph()
    graph.parse(url)

    # Execute the query
    subset = graph.query(QUERY_ONTOLOGY_SUBSET)

    # Serialize the result into a new Turtle file
    subset_graph = rdflib.Graph()
    subset_graph.bind("know", "https://know.dev/")
    for triple in subset:
        subset_graph.add(triple)

    return subset_graph.serialize(format="turtle")

# Load the KNOW ontology
know_ontology = load_ontology(KNOW_ONTOLOGY_URL)

# Query Ollama API
def query_ollama(model: str, prompt: str):
    llm = Ollama(model=model, request_timeout=360.0)
    return llm.complete(prompt).text

# Prompt Templates
PROMPT_TEMPLATE_CAPTURE = """
You are a system designed to extract knowledge from user inputs and map it to the ontology provided.

Ontology:
{ontology}

User Input: {user_input}

Output the extracted knowledge as RDF triples in Turtle format.
Capture only one knowledge at time.
Please always use only `know` as prefix in output.

For example:
User input: Nathan loved to take his sister, Donna, with him whenever he went shopping.

Your output:
@prefix know: <https://know.dev/> .

know:Nathan a know:Person ;
    know:name "Nathan" ;
    know:sister know:Donna .

know:Donna a know:Person ;
    know:name "Donna".

Ensure the format adheres to the ontology structure.
Please output only RDF triples, no other text, note or explanation.
"""

PROMPT_TEMPLATE_CLASSIFY = """
You are an intelligent assistant designed to classify user input into one of two categories based on the given ontology and context.

Your task is to decide the most suitable classification for the user's input:
1. "Capture Knowledge" if the input provides new information that can be mapped to the ontology (e.g., introducing new relationships, facts, or events related to the ontology).
2. "Retrieve Knowledge" if the input is a query or question that requires retrieving information already stored in the ontology.

Do not provide any additional text—only output one of the two classifications: "Capture Knowledge" or "Retrieve Knowledge."

Examples:

Ontology: 
Ontology about people, and their relatives.

Example 1:
User Input: "Nathan is Donna's brother, and they often go shopping together."
Classification: "Capture Knowledge"

Example 2:
User Input: "Who are Nathan's siblings?"
Classification: "Retrieve Knowledge"

User Input: {user_input}
Classification:
"""

def classify_input(user_input, ontology, model=MODEL_NAME):
    formatted_prompt = PROMPT_TEMPLATE_CLASSIFY.format(ontology=ontology, user_input=user_input)
    classification = query_ollama(model, formatted_prompt)
    return classification

def capture_knowledge(user_input, ontology, model=MODEL_NAME):
    formatted_prompt = PROMPT_TEMPLATE_CAPTURE.format(ontology=ontology, user_input=user_input)
    rdf_output = query_ollama(model, formatted_prompt)
    return rdf_output

def store_knowledge(rdf_triples, turtle_file=TURTLE_FILE):
    graph1 = rdflib.Graph()
    try:
        # Load existing triples if the file exists
        graph1.parse(turtle_file, format="turtle")
    except FileNotFoundError:
        pass
    # Add new triples
    graph2 = rdflib.Graph()
    graph2.parse(data=rdf_triples, format="turtle")

    graph = graph1 + graph2
    # Save updated graph back to file
    graph.serialize(turtle_file, format="turtle")
    print("Knowledge stored successfully!")


PROMPT_TEMPLATE_SPARQL = """
You are an expert in generating valid SPARQL queries. Your job is to generate a SPARQL query to retrieve specific knowledge from an ontology based on the user's input.

Ontology:
---
{ontology}
---

Requirements:
1. Use valid SPARQL syntax.
2. Include necessary prefixes.
3. Ensure the query retrieves the correct `?subject`, `?predicate`, and `?object` based on the user's input.
4. Do not provide any additional text except SPARQL query.
5. Never use backticks.

Examples:
User Input: "Who is Johnny's sister?"

SPARQL Query:

PREFIX : <https://know.dev/>
SELECT *
WHERE {{
  ?s a :Person ;
     ?p ?o .
    FILTER (?s = :Johnny && ?p = :sister)
}}

User Input: "{user_input}"
SPARQL Query:
"""

def generate_sparql_query(user_input, ontology, model=MODEL_NAME):
    """
    Generate a SPARQL query using the LLM based on the user's input and the ontology.
    """
    prompt = PROMPT_TEMPLATE_SPARQL.format(user_input=user_input, ontology=ontology)
    sparql_query = query_ollama(model, prompt).strip()
    print(f'Generated SPARQL Query:\n{sparql_query}')

    try:
        # Test parsing the query with rdflib
        rdflib.plugins.sparql.processor.prepareQuery(sparql_query)
    except Exception as e:
        raise e

    return sparql_query


def retrieve_knowledge(user_input, ontology, turtle_file=TURTLE_FILE, model=MODEL_NAME):
    """
    Retrieve knowledge by generating a SPARQL query using LLM and executing it on the Turtle file.
    """
    graph = rdflib.Graph()
    graph.parse(turtle_file, format="turtle")
    
    # Generate SPARQL query using LLM
    sparql_query = generate_sparql_query(user_input, ontology, model=model)
    
    # Execute the SPARQL query
    try:
        results = graph.query(sparql_query)
        results_graph = rdflib.Graph()
        results_graph.bind("know", "https://know.dev/")
        for triple in results:
            if triple:
                # return first result.
                results_graph.add(triple)
                return results_graph.serialize(format="turtle")
            else:
                return "No relevant knowledge found."

    except Exception as e:
        return f"Error executing SPARQL query: {e}"


if __name__ == "__main__":
    print("Welcome to the Ontology-Guided Knowledge Capture System using Ollama!")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter your input: ")
        if user_input.lower() == "exit":
            break
        
        classification = classify_input(user_input, know_ontology, model=MODEL_NAME)
        print(f"Classification: {classification}")

        if 'Capture Knowledge' in classification:
            rdf_triples = capture_knowledge(user_input, know_ontology, model=MODEL_NAME)
            print("\nExtracted Knowledge (in Turtle format):\n")
            print(rdf_triples)
            store_knowledge(rdf_triples, turtle_file=TURTLE_FILE)
        elif 'Retrieve Knowledge' in classification:
            response = retrieve_knowledge(user_input, know_ontology, turtle_file=TURTLE_FILE)
            print("\nRetrieved Knowledge:\n")
            print(response)
        else:
            print("\nCould not classify the input. Please try again.")
