# 

```
git clone https://github.com/modelcontextprotocol/servers official-mcp-servers
uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul
uv run scripts/mcp/client.py "uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul"
```