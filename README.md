# 

```
git clone https://github.com/modelcontextprotocol/servers official-mcp-servers
uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul
uv run scripts/mcp/client.py "uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul"


uv run scripts/mcp/client.py "npx -y @modelcontextprotocol/server-filesystem /home/lsw91/Workspace/ai-playground-1"
```



```
npm install -g local-ssl-proxy

# 기본 설정
local-ssl-proxy --source 8443 --target 8501
local-ssl-proxy --source 8443 --target 8501 --cert localhost.pem --key localhost-key.pem
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout localhost-key.pem -out localhost.pem
```