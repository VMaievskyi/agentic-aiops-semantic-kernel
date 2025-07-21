from utils.Config import config
from azure.ai.agents.models import McpTool



class Mcps:
       
    
    async def mcps(self) -> McpTool:
        return McpTool(
                                server_label=config.mcp_server_label,
                                server_url=config.mcp_server_url,
                                allowed_tools=[],  # Optional: specify allowed tools
                        )
    
mcp = Mcps()