"""
Mock out langchain internals so tests run without requiring a specific
langchain version to be installed. The real production code uses langchain;
these tests focus on the routing, deduplication, and fallback logic.
"""
import sys
from unittest.mock import MagicMock

# Stub langchain modules before any src import triggers them
_LANGCHAIN_MODS = [
    "langchain",
    "langchain.agents",
    "langchain.tools",
    "langchain.prompts",
    "langchain.schema",
    "langchain.memory",
    "langchain.callbacks",
    "langchain_openai",
    "langchain_community",
    "langchain_community.tools",
]

for mod in _LANGCHAIN_MODS:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Patch the specific classes that research_agent.py imports at module level
from unittest.mock import MagicMock

langchain_agents = sys.modules["langchain.agents"]
langchain_agents.AgentExecutor = MagicMock
langchain_agents.create_openai_tools_agent = MagicMock(return_value=MagicMock())

sys.modules["langchain.tools"].Tool = MagicMock
sys.modules["langchain.prompts"].ChatPromptTemplate = MagicMock
sys.modules["langchain.prompts"].MessagesPlaceholder = MagicMock
sys.modules["langchain.schema"].BaseMessage = object
sys.modules["langchain.schema"].HumanMessage = MagicMock
sys.modules["langchain.schema"].AIMessage = MagicMock
sys.modules["langchain.schema"].SystemMessage = MagicMock
sys.modules["langchain.memory"].ConversationBufferWindowMemory = MagicMock
sys.modules["langchain.callbacks"].get_openai_callback = MagicMock
sys.modules["langchain_openai"].ChatOpenAI = MagicMock
sys.modules["langchain_community.tools"].DuckDuckGoSearchRun = MagicMock
