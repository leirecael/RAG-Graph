#https://medium.com/@dlaytonj2/openai-agent-sdk-hooks-and-usage-3f4226f34f39

from agents import Agent, RunContextWrapper, RunHooks, Tool, Usage

class MyHooks(RunHooks):
    
    def __init__(self):
        self.event_counter = 0
        # added - keep track of total input and output tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0 

    def reset(self) -> None:
        self.event_counter = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _usage_to_str(self, usage: Usage) -> str:
        return f"{usage.requests} requests, {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, {usage.total_tokens} total tokens"

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1                         
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        #print(f"### {self.event_counter}: Agent {agent.name} started. Usage: {self._usage_to_str(context.usage)}")

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        
        #print(f"### {self.event_counter}: Agent {agent.name} ended with output {output}. Usage: {self._usage_to_str(context.usage)}")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        #print(f"### {self.event_counter}: Tool {tool.name} started. Usage: {self._usage_to_str(context.usage)}")

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
       
        #print(f"### {self.event_counter}: Tool {tool.name} ended with result {result}. Usage: {self._usage_to_str(context.usage)}")

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        self.event_counter += 1
        self.total_input_tokens += context.usage.input_tokens
        self.total_output_tokens += context.usage.output_tokens
        #print(f"### {self.event_counter}: Handoff from {from_agent.name} to {to_agent.name}. Usage: {self._usage_to_str(context.usage)}")