from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Create a couple of Global Variables
TEMPERATURE = 0.2
MAX_TOKENS = 512

# GPT-5.6 Luna. reasoning_effort="none" is required for /v1/chat/completions
# plus tool calling — the same pin used in ISOM 352.
openai_gpt56_luna = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-5.6-luna",
    max_tokens=300,
    reasoning_effort="none",
)

openai_gpt56_luna_json = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-5.6-luna",
    max_tokens=300,
    reasoning_effort="none",
    model_kwargs={"response_format": {"type": "json_object"}},
)

openai_gpt56_luna_router = ChatOpenAI(
    temperature=0.1,
    model="gpt-5.6-luna",
    max_tokens=50,
    reasoning_effort="none",
)

# Names the rest of the app still imports.
openai_gpt4o_mini = openai_gpt56_luna
openai_4o_mini_json = openai_gpt56_luna_json
openai_gpt4 = openai_gpt56_luna_router
openai_gpt4o = openai_gpt56_luna

claude_haiku = ChatAnthropic(
        model='claude-haiku-4-5',
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )
