import logging
import sys
from semantic_kernel.agents import (
    Agent,
    ChatHistoryAgentThread,
    ChatCompletionAgent,
    MagenticOrchestration,
    StandardMagenticManager
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent
from utils.Config import config
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from semantic_kernel.agents.runtime import InProcessRuntime

from azure.ai.projects import AIProjectClient
from semantic_kernel.agents import AzureAIAgent
from azure.core.credentials import AzureKeyCredential


token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

#1. Preparation for connecting with Azure AI Foundry
project_client =  AIProjectClient(
    credential=AzureKeyCredential(config.azure_agent_api_key),
    
    endpoint="https://gen-bi-foundry.services.ai.azure.com/api/projects/gen-bi-foundry")

# Configure logging to output to console with detailed info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("Agents")

#2. Declaring agents - mix of AI foundry + in memory
class Agents:
    """
    Class to manage agents using MagenticOrchestration.
    """

    def __init__(self) -> None:
        logger.debug("Initializing Agents class.")
        self.magentic_orchestration = None
        self.runtime = InProcessRuntime()
        self.thread: ChatHistoryAgentThread = None

        logger.debug(f"Environment: {config.environment}")

        #3. In memory agent - chat, handles messages from USer, AzureChatCompletion - required for magentic orchestration
        if config.environment == "dev":
            logger.debug("Using API key authentication for AzureChatCompletion.")
            self.chat_service = AzureChatCompletion(
                deployment_name=config.azure_openai_deployment,
                api_version=config.azure_openai_api_version,
                endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key
            )
        else:
            logger.debug("Using Azure AD token authentication for AzureChatCompletion.")
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                config.llm_model_scope
            )
            self.chat_service = AzureChatCompletion(
                deployment_name=config.azure_openai_deployment,
                api_version=config.azure_openai_api_version,
                endpoint=config.azure_openai_endpoint,
                ad_token_provider=token_provider
            )
        

    async def agents(self) -> list[Agent]:
        logger.debug("Building agent list.")


        #4. Get sql generator from ai foundry
        sql_def = project_client.agents.get_agent("asst_WCxADWCUJPjqgH5w5t8oT4el")
        logger.debug("SQL genertor is created:")

        #5. in memory agent with agent type required for magentic orchestration
        chat_agent = ChatCompletionAgent(
            name="chat_specialist",
            service=self.chat_service,
            instructions="You are bi expert, you are supposed to delegate user requests to other agents to fulfil.",
            description="",
            plugins=[],
        )


        #6. Get sql validator ai foundry
        validator_def = project_client.agents.get_agent("asst_tsRLnPXKWe5wjXvuph2mA0R9")
        logger.debug("SQL validator is created:")



        logger.debug("Agents creating: sql_agent, validator_agent")

        #7. Wrap ai foundry agents with type required for orchestration
        sql_agent = AzureAIAgent(client=project_client, definition=sql_def)
        validator_agent = AzureAIAgent(client=project_client, definition=validator_def)

        return [chat_agent, sql_agent, validator_agent]
        
    async def run_task(self, payload: str) -> None:
        """
        Runs the agent's task with the provided payload.
        """
        logger.info(f"Starting run_task with payload: {payload}")
        try:
            magentic_orchestration = MagenticOrchestration(
                members=await self.agents(),
                manager=StandardMagenticManager(chat_completion_service=self.chat_service),
                agent_response_callback=self._agent_response_callback
            )

            logger.debug("Starting InProcessRuntime.")
            self.runtime.start()

            logger.debug("Invoking MagenticOrchestration.")
            orchestration_result = await magentic_orchestration.invoke(
                task=payload,
                runtime=self.runtime
            )

            logger.debug("Awaiting orchestration result.")
            value = await orchestration_result.get()
            logger.info(f"Final result: {value}")

            logger.debug("Stopping runtime when idle.")
            await self.runtime.stop_when_idle()
            logger.info("run_task completed successfully.")
        except Exception as e:
            logger.error(f"Exception in run_task: {e}", exc_info=True)

    @staticmethod
    def _agent_response_callback(message: ChatMessageContent) -> None:
        """Observer function to print the messages from the agents and manager."""
        logger = logging.getLogger("Agents.Callback")
        logger.info(f"Agent/Manager Message - **{message.name}**: {message.content}")