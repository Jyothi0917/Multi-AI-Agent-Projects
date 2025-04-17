import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.chat_models import ChatLiteLLM  # Import ChatLiteLLM from LangChain Community

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face with API key
huggingface_token = os.getenv("HUGGINGFACE_API_KEY")
if huggingface_token is None:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables.")
login(token=huggingface_token)

# Set up the LLM using Hugging Face's Zephyr model with correct provider configuration
llm = ChatLiteLLM(
    model="huggingface/HuggingFaceH4/zephyr-7b-beta",  # Correct model ID format
    api_key=huggingface_token,  # Use the API key directly here
    provider="huggingface",  # Specify the provider explicitly
    temperature=0.7,
    max_tokens=256
)

# Skin Profile Agent with user input prompts
skin_profile_agent = Agent(
    role="Skin Profile Expert",
    goal="Analyze user's skin type, concerns, and environment based on inputs.",
    backstory="You're a senior dermatologist trained to identify skin characteristics and external factors that affect the skin.",
    allow_delegation=False,
    llm=llm  # Pass the LangChain LLM to the agent
)

def get_skin_profile_input():
    print("Welcome to the Skin Profile Analyzer!\nPlease answer the following questions to help us understand your skin better:\n")

    # Collect user inputs
    skin_type = input("1. What is your skin type? (e.g., oily, dry, combination, sensitive): ")
    skin_concerns = input("2. What are your skin concerns? (e.g., acne, dullness, wrinkles, dark spots): ")
    environment = input("3. What is your environment like? (e.g., humid, dry, polluted, sunny, etc.): ")

    return {
        "skin_type": skin_type,
        "skin_concerns": skin_concerns,
        "environment": environment
    }

# Skin Profiling Task with dynamic user input
def create_skin_profile_task():
    user_input = get_skin_profile_input()

    skin_profile_task = Task(
        name="Analyze Skin Profile",
        description=f"Analyze the user's skin type: {user_input['skin_type']}, concerns: {user_input['skin_concerns']}, and environment: {user_input['environment']}.",
        agent=skin_profile_agent,
        expected_output="A detailed analysis of how the user's skin type and environment affect their skin.",
    )
    return skin_profile_task

# Routine Generator Agent
routine_generator_agent = Agent(
    role="Routine Generator Specialist",
    goal="Create a tailored 3-4 step skincare routine",
    backstory="You are an expert skin-care routine designer who creates routines based on expert dermatologist input.",
    allow_delegation=True,
    llm=llm  # Pass the LangChain LLM to the agent
)

# Product Suggestion Agent
product_agent = Agent(
    role="Product Advisor",
    goal="Suggest general types of products suitable for the skincare routine steps",
    backstory="You're a product knowledge expert who knows what types of ingredients or products suit different skin types and concerns.",
    allow_delegation=False,
    llm=llm  # Pass the LangChain LLM to the agent
)

# Create the tasks and crew
skin_profile_task = create_skin_profile_task()

routine_task = Task(
    name="Create Routine",
    description="Based on skin analysis, generate a morning and night skincare routine with 3-4 steps.",
    agent=routine_generator_agent,
    expected_output="A brief step-by-step skincare routine tailored to the user's skin profile.",
)

product_task = Task(
    name="Recommend Products",
    description="Based on the routine, recommend specific products for each step. Make sure they’re safe for sensitive skin.",
    agent=product_agent,
    expected_output="List of few specific product suggestions for each step in the skincare routine.",
)

# Create a Crew with the three agents and their tasks
skincare_crew = Crew(
    agents=[skin_profile_agent, routine_generator_agent, product_agent],
    tasks=[skin_profile_task, routine_task, product_task],
    verbose=True
)

# Run the Crew AI agents
result = skincare_crew.kickoff()
print("\n\n✅ FINAL RESULT:\n", result)
