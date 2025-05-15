# 

```
git clone https://github.com/modelcontextprotocol/servers official-mcp-servers
uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul
uv run scripts/mcp/client.py "uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul"


uv run scripts/mcp/client.py "npx -y @modelcontextprotocol/server-filesystem /home/lsw91/Workspace/ai-playground-1"
```