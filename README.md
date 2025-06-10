# 

```
git clone https://github.com/modelcontextprotocol/servers official-mcp-servers
uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul
uv run scripts/mcp/client.py "uvx --from official-mcp-servers/src/time/ mcp-server-time --local-timezone Asia/Seoul"


uv run scripts/mcp/client.py "npx -y @modelcontextprotocol/server-filesystem /home/lsw91/Workspace/ai-playground-1"
```



```
npm install -g local-ssl-proxy

# HTTPS 인증서 생성 (SAN 포함)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout localhost-key.pem -out localhost.pem -subj "/C=KR/ST=Seoul/L=Seoul/O=Development/CN=localhost" -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1"

# SSL 프록시 실행
local-ssl-proxy --source 8443 --target 8501 --cert localhost.pem --key localhost-key.pem

# 크롬에서 "보안연결이 사용되지 않았습니다" 해결 방법:

## 방법 1: 크롬에서 "고급" -> "안전하지 않음(localhost)(안전하지 않음)으로 이동" 클릭

## 방법 2: 인증서를 시스템에 추가 (Ubuntu/Linux)
# sudo cp localhost.pem /usr/local/share/ca-certificates/localhost.crt
# sudo update-ca-certificates

## 방법 3: 크롬 플래그 사용 (개발용)
# chrome://flags/#allow-insecure-localhost 에서 "Enabled" 설정

## 방법 4: 브라우저에서 직접 인증서 신뢰 설정
# 1. https://localhost:8443 접속
# 2. "고급" 클릭
# 3. "localhost(안전하지 않음)으로 이동" 클릭
# 4. 또는 인증서 오류 화면에서 "이 사이트 계속 사용" 선택
```

# 커스텀 포트
```
uv run streamlit run scripts/multimodal/streamlit-multimodal-chatbot.py --server.port 47886
```


```
[
    {
        "name": "sequential-thinking",
        "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
      ],
        "enabled": true
    },
 {
    "name": "filesystem",
    "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/lsw91/ai-data-storage"
      ]
 },
    {
        "name": "memory",
        "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-memory"
      ],
        "enabled": true
    },
    {
      "name": "chroma",
      "command": "uvx",
      "args": [
          "chroma-mcp",
          "--client-type",
          "persistent",
          "--data-dir",
          "/home/lsw91/ai-data-storage"
      ]
    },
    {
        "name": "notionApi",
        "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
      "env": {
        "OPENAPI_MCP_HEADERS": "{\"Authorization\": \"Bearer ntn_***\", \"Notion-Version\": \"2022-06-28\" }"
      },
        "enabled": true
    },
    {
       "name":  "playwright",
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ]
    }
]
```