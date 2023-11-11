from typing import Dict, Any, Tuple, Generator
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.memory import ConversationBufferMemory
from openai import BadRequestError
from pathlib import Path

MODEL_NAME = "gpt-4"


class ConversationalAgent:
    def __init__(self, working_directory: Path) -> None:
        self.intermediate_steps = []
        self.working_directory = working_directory
        self.tools = self.setup_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
        llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in self.tools])

        self.agent_chain = self.setup_chain(llm_with_tools, is_fallback=False)
        self.fallback_chain = self.setup_chain(llm, is_fallback=True)

    def setup_tools(self) -> Dict[str, Any]:
        return FileManagementToolkit(
            root_dir=str(self.working_directory.name),
            selected_tools=["read_file", "write_file", "list_directory"]
        ).get_tools()

    def setup_chain(self, llm: ChatOpenAI, is_fallback: bool) -> Any:
        prompt = self.create_prompt(is_fallback)
        assigns = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
            "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"]
        }
        if is_fallback:
            return assigns | prompt | llm | StrOutputParser()
        else:
            return assigns | prompt | llm | OpenAIFunctionsAgentOutputParser()

    def create_prompt(self, is_fallback: bool) -> ChatPromptTemplate:
        system_message = ("You are the AI that tells the user what the error is in plain Japanese. "
                          "Since the error occurs at the end of the step, you must guess from the process flow "
                          "and the error message, and communicate the error message to the user in an easy-to-understand manner.") \
            if is_fallback else "You are a useful assistant."
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    async def run(self, input_message: str) -> Generator[Tuple[str, bool], None, None]:
        while True:
            message, is_final = await self.process_step(input_message)
            yield {"message": message, "is_final": is_final}
            if is_final:
                self.finish_process(input_message, message)
                break

    async def process_step(self, input_message: str) -> Tuple[str, bool]:
        try:
            output = await self.agent_chain.ainvoke({
                "input": input_message,
                "intermediate_steps": self.intermediate_steps
            })
            if isinstance(output, AgentFinish):
                return output.return_values["output"], True
            else:
                observation = self.tool_execute(output.tool, output.tool_input)
                message = self.format_tool_log(output.tool, output.tool_input, observation)
                self.intermediate_steps.append((output, observation))
                return message, False
        except BadRequestError as error:
            return self.handle_error(error), True

    def handle_error(self, error: BadRequestError) -> str:
        # コンテキスト長あふれの可能性もあるため、最後のステップのツール実行結果を空にする
        self.intermediate_steps[-1] = (self.intermediate_steps[-1][0], "")
        return self.fallback_chain.invoke({
            "input": error.response.json()["error"]["message"],
            "intermediate_steps": self.intermediate_steps
        })

    def tool_execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        read_file, write_file, list_directory = self.tools
        tool = {
            "read_file": read_file,
            "write_file": write_file,
            "list_directory": list_directory,
        }[tool_name]
        if tool:
            return tool.run(tool_input)
        else:
            raise ValueError(f"対応していないツールが選択されました: {tool_name}")

    def format_tool_log(self, tool_name: str, tool_input: Dict[str, Any], observation: str) -> str:
        if tool_name == "read_file":
            result = (f"{tool_input['file_path']}を読み込みました。\n"
                      f"文字数：{len(observation)}\n")
        else:
            result = (f"{observation}\n")

        return (f"ツール選択\n"
                f"{tool_name}, {str(tool_input)}\n\n"
                f"実行結果\n"
                f"{result}")

    def finish_process(self, input_message: str, output_message: str) -> None:
        self.memory.save_context({"input": input_message}, {"output": output_message})
        self.intermediate_steps = []
