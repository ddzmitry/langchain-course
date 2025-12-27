# Migration from ReAct Agent to Function Calling Agent

## Overview
This document outlines the migration process from a ReAct-based agent to a function calling agent using LangChain's `.bind_tools()` method with manual message handling.

## Key Differences

### ReAct Agent (Before)
- **Text-based parsing**: Uses `ReActSingleInputOutputParser` to extract actions from text
- **Format instructions**: Requires explicit "Thought/Action/Action Input/Observation" format in prompts
- **String manipulation**: Actions and inputs parsed from LLM text output using regex
- **Manual tracking**: Developer manually formats observation logs
- **Fragile**: Prone to parsing errors if LLM doesn't follow exact format
- **Stop sequences**: Uses `stop=["\nObservation"]` to control output

### Function Calling Agent (After)
- **Structured outputs**: LLM returns tool calls as structured JSON
- **Message-based**: Uses `HumanMessage`, `AIMessage`, and `ToolMessage` objects
- **Type-safe**: Tool calls have guaranteed structure with `id`, `name`, `args`
- **Automatic tracking**: `tool_call_id` links responses to requests
- **Reliable**: No parsing errors, schema-validated by OpenAI
- **Native support**: No stop sequences needed

## Migration Steps

### Step 1: Update Imports

**Before (ReAct):**
```python
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.schema import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
```

**After (Function Calling):**
```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool, BaseTool
```

### Step 2: Initialize LLM and Bind Tools

**Before (ReAct):**
```python
llm = ChatOpenAI(
    temperature=0,
    stop=["\nObservation", "Observation"],  # Stop sequences for parsing
)
# Tools are passed to prompt, not LLM
```

**After (Function Calling):**
```python
llm = ChatOpenAI(temperature=0)
llm_with_tools = llm.bind_tools(tools)  # KEY: Bind tools to LLM
```

### Step 3: Remove Complex Prompts

**Before (ReAct):**
```python
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools),
    tool_names="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
)
```

**After (Function Calling):**
```python
# No prompt needed! Just start with a message
messages = [HumanMessage(content="What is the length of 'strawberry' in characters?")]
```

### Step 4: Replace Manual Loop with Message Loop

**Before (ReAct):**
```python
intermediate_steps = []
agent_step = ""

while not isinstance(agent_step, AgentFinish):
    agent_step = agent.invoke({
        "input": "What is the length of 'strawberry' in characters?",
        "agent_scratchpad": intermediate_steps,
    })

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_input = agent_step.tool_input
        tool_to_call = find_tool_by_name(tools, tool_name)
        observation = tool_to_call.func(tool_input)
        intermediate_steps.append([agent_step, str(observation)])

if isinstance(agent_step, AgentFinish):
    print(f"Final answer: {agent_step.return_values['output']}")
```

**After (Function Calling):**
```python
messages = [HumanMessage(content="What is the length of 'strawberry' in characters?")]

while True:
    ai_message = llm_with_tools.invoke(messages)

    tool_calls = getattr(ai_message, "tool_calls", None) or []

    if len(tool_calls) > 0:
        messages.append(ai_message)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            tool_to_use = find_tool_by_name(tools, tool_name)
            observation = tool_to_use.invoke(tool_args)

            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )
        continue

    # No tool calls = final answer
    print(ai_message.content)
    break
```

### Step 5: Remove Custom Parsing Functions

**Before (ReAct):**
```python
def format_log_to_string(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
```

**After (Function Calling):**
```python
# Not needed! Messages handle conversation history automatically
```

## Complete Before/After Comparison

### ReAct Agent (Before)
```python
# Complex setup with prompts and parsers
tools = [get_text_length]
template = """...(long prompt with format instructions)..."""
prompt = PromptTemplate.from_template(template=template).partial(...)
llm = ChatOpenAI(temperature=0, stop=["\nObservation", "Observation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_string(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)

# Manual loop with type checking
intermediate_steps = []
agent_step = ""
while not isinstance(agent_step, AgentFinish):
    agent_step = agent.invoke({
        "input": "What is the length of 'strawberry'?",
        "agent_scratchpad": intermediate_steps,
    })
    if isinstance(agent_step, AgentAction):
        # Parse and execute tool
        # ...
```

### Function Calling Agent (After)
```python
# Simple setup
tools = [get_text_length]
llm = ChatOpenAI(temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Clean message-based loop
messages = [HumanMessage(content="What is the length of 'strawberry'?")]
while True:
    ai_message = llm_with_tools.invoke(messages)
    tool_calls = getattr(ai_message, "tool_calls", None) or []

    if len(tool_calls) > 0:
        messages.append(ai_message)
        for tool_call in tool_calls:
            tool_to_use = find_tool_by_name(tools, tool_call.get("name"))
            observation = tool_to_use.invoke(tool_call.get("args", {}))
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call.get("id"))
            )
        continue

    print(ai_message.content)
    break
```

## Benefits of Migration

### 1. Code Simplicity
- **Before**: ~150 lines of code with prompt templates, parsers, and format functions
- **After**: ~50 lines of clean, readable code

### 2. Reliability
- **Before**: String parsing can fail if LLM outputs slightly wrong format
- **After**: Structured JSON validated by OpenAI, guaranteed correct format

### 3. Maintainability
- **Before**: Changing prompt format requires updating parser logic
- **After**: Just modify tools list, no parser changes needed

### 4. Debugging
- **Before**: Must print and inspect text strings to understand agent behavior
- **After**: Clear message objects with structured data

### 5. Extensibility
- **Before**: Adding new tools requires updating prompt template
- **After**: Just add to tools list, `.bind_tools()` handles the rest

## How .bind_tools() Works

1. **Schema Generation**: Converts Python function signatures to JSON schemas
   ```python
   @tool
   def get_text_length(text: str) -> int:
       """Returns the length of a text by characters"""
       return len(text)

   # Automatically becomes:
   {
       "name": "get_text_length",
       "description": "Returns the length of a text by characters",
       "parameters": {
           "type": "object",
           "properties": {
               "text": {"type": "string"}
           },
           "required": ["text"]
       }
   }
   ```

2. **Tool Calling**: LLM receives schemas and can request tool invocations
   ```python
   # LLM response structure:
   {
       "content": "",  # Empty if using tools
       "tool_calls": [
           {
               "id": "call_abc123",
               "type": "function",
               "name": "get_text_length",
               "args": {"text": "strawberry"}
           }
       ]
   }
   ```

3. **Result Handling**: ToolMessage connects result back to request
   ```python
   ToolMessage(
       content="10",  # The result
       tool_call_id="call_abc123"  # Links to original request
   )
   ```

## Testing Strategy

The migration maintains the same functionality while improving code quality:

1. **Tool behavior**: Same tools, same results
2. **Conversation flow**: Still multi-turn, just cleaner
3. **Error handling**: Better error messages from structured data
4. **Edge cases**: No more parsing edge cases to worry about

## Common Pitfalls

1. **Forgetting to append ai_message**: Must add AI's tool call message before adding tool results
   ```python
   if len(tool_calls) > 0:
       messages.append(ai_message)  # Don't forget this!
       # ... execute tools ...
   ```

2. **Missing tool_call_id**: ToolMessage must include the id to link to request
   ```python
   ToolMessage(
       content=str(observation),
       tool_call_id=tool_call.get("id")  # Required!
   )
   ```

3. **Not continuing loop**: After adding tool results, must `continue` to let LLM process them
   ```python
   messages.append(ToolMessage(...))
   continue  # Let LLM see the results!
   ```

## Conclusion

Function calling with `.bind_tools()` is:
- **Simpler**: Less code, clearer intent
- **More reliable**: No parsing errors
- **Easier to maintain**: Fewer moving parts
- **Better supported**: Native model capability

The manual message handling approach (vs using AgentExecutor) provides:
- **Transparency**: See exactly what's happening
- **Control**: Customize behavior at each step
- **Learning**: Understand the mechanics deeply
- **Flexibility**: Easy to add custom logic
