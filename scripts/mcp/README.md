# MCP Multimodal Chatbot

멀티모달 기능(텍스트, 이미지, PDF)과 Model Context Protocol(MCP) 도구를 지원하는 Streamlit 기반 AI 챗봇입니다.

## 기능

- 🎨 **멀티모달 지원**: 텍스트, 이미지, PDF 파일 처리
- 🔧 **MCP 통합**: 여러 MCP 서버를 실시간으로 연결/해제 가능
- 💰 **비용 추적**: 토큰 사용량 및 API 비용 실시간 모니터링
- 🛑 **스트리밍 제어**: 응답 생성 중 중단 가능
- 📊 **세션 통계**: 전체 세션의 토큰 사용량 및 비용 추적

## 설치

1. 필요한 패키지 설치:
```bash
pip install streamlit anthropic mcp python-dotenv
```

2. MCP 서버를 위한 `uv` 설치 (uvx 명령어 사용을 위해):
```bash
# Python의 pip를 통해 설치
pip install uv

# 또는 시스템 패키지 매니저 사용
# macOS
brew install uv

# Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. `.env` 파일에 Anthropic API 키 설정:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## 사용법

### 실행

```bash
streamlit run scripts/mcp/streamlit-mcp-multimodal-chatbot.py
```

### MCP 서버 설정

1. 사이드바의 "MCP Configuration" 섹션에서 JSON 형식으로 MCP 서버 설정을 입력합니다.

예시:
```json
[
    {
        "name": "time",
        "command": "uvx",
        "args": [
            "--from",
            "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/time",
            "mcp-server-time",
            "--local-timezone",
            "Asia/Seoul"
        ],
        "enabled": true
    },
    {
        "name": "weather",
        "command": "uvx",
        "args": [
            "--from",
            "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/weather",
            "mcp-server-weather",
            "--api-key",
            "YOUR_OPENWEATHER_API_KEY"
        ],
        "enabled": true
    }
]
```

2. "➕ Add Servers" 버튼을 클릭하여 서버를 추가합니다.

3. 각 서버의 토글 스위치로 활성화/비활성화를 설정합니다.

4. "🔄 Apply Changes" 버튼을 클릭하여 변경사항을 적용합니다.

### MCP 서버 예시

#### Time Server
현재 시간을 가져오는 도구를 제공합니다:
```json
{
    "name": "time",
    "command": "uvx",
    "args": [
        "--from",
        "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/time",
        "mcp-server-time",
        "--local-timezone",
        "Asia/Seoul"
    ],
    "enabled": true
}
```

사용 가능한 도구:
- `get_current_time`: 특정 타임존의 현재 시간 조회
- `convert_time`: 시간대 간 시간 변환

#### Weather Server
날씨 정보를 가져오는 도구를 제공합니다 (OpenWeather API 키 필요):
```json
{
    "name": "weather",
    "command": "uvx",
    "args": [
        "--from",
        "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/weather",
        "mcp-server-weather",
        "--api-key",
        "YOUR_API_KEY_HERE"
    ],
    "enabled": true
}
```

사용 가능한 도구:
- `get_weather`: 특정 도시의 날씨 정보 조회

#### Git Server
Git 저장소 작업을 위한 도구를 제공합니다:
```json
{
    "name": "git",
    "command": "uvx",
    "args": [
        "--from",
        "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/git",
        "mcp-server-git",
        "--repository",
        "/path/to/your/repo"
    ],
    "enabled": true
}
```

#### Filesystem Server
파일 읽기/쓰기 도구를 제공합니다:
```json
{
    "name": "filesystem",
    "command": "uvx",
    "args": [
        "--from",
        "git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/filesystem",
        "mcp-server-filesystem",
        "/path/to/allowed/directory"
    ],
    "enabled": true
}
```

#### 커스텀 MCP Server
자신만의 MCP 서버를 만들어 사용할 수 있습니다:
```json
{
    "name": "custom_server",
    "command": "python",
    "args": ["path/to/your/mcp_server.py"],
    "env": {
        "CUSTOM_VAR": "value"
    },
    "enabled": true
}
```

### 사용 예시

1. **시간 확인하기**: 
   - Time MCP 서버를 추가하고 활성화
   - "현재 시간이 뭐야?" 또는 "뉴욕 시간 알려줘" 라고 질문
   - "서울 시간 14:30은 뉴욕으로 몇 시야?" 같은 시간 변환도 가능

2. **날씨 확인하기**:
   - Weather MCP 서버를 추가하고 활성화 (API 키 필요)
   - "서울 날씨 어때?" 라고 질문

3. **파일 작업하기**:
   - Filesystem MCP 서버를 추가하고 활성화
   - "test.txt 파일에 'Hello World'라고 써줘" 라고 요청
   - "test.txt 파일 내용 보여줘" 라고 질문

4. **이미지와 함께 질문하기**:
   - 이미지를 업로드하고 "이 이미지에 뭐가 있어?" 라고 질문
   - PDF를 업로드하고 "이 문서 요약해줘" 라고 요청

## 주의사항

- MCP 서버가 실행되려면 해당 서버의 실행 환경이 준비되어 있어야 합니다.
- `uvx`를 사용하는 경우 `uv`가 설치되어 있어야 합니다.
- Git 기반 설치는 인터넷 연결이 필요하며, 첫 실행 시 패키지를 다운로드합니다.
- 파일시스템 접근 권한이 필요한 서버의 경우 적절한 권한 설정이 필요합니다.
- 일부 MCP 서버는 API 키나 추가 설정이 필요할 수 있습니다.

## 문제 해결

### MCP 서버 연결 실패
- 서버 command와 args가 올바른지 확인
- 필요한 의존성이 설치되어 있는지 확인 (특히 `uv`)
- 터미널에서 직접 명령어를 실행해보며 오류 확인
  ```bash
  # 예시: Time 서버 직접 실행
  uvx --from git+https://github.com/modelcontextprotocol/servers.git#subdirectory=src/time mcp-server-time
  ```

### 도구 호출 실패
- MCP 서버가 정상적으로 연결되어 있는지 확인
- 도구 이름과 파라미터가 올바른지 확인
- 서버 로그에서 오류 메시지 확인

### uvx 명령어를 찾을 수 없음
- `uv`가 설치되어 있는지 확인
- PATH에 `uv`가 추가되어 있는지 확인
- 전체 경로를 사용해보세요:
  ```json
  {
      "name": "time",
      "command": "/home/username/.local/bin/uvx",
      "args": ["--from", "git+...", "mcp-server-time"],
      "enabled": true
  }
  ```

### 타임존 오류
- IANA 표준 타임존 이름을 사용하세요 (예: 'Asia/Seoul', 'America/New_York')
- 'KST', 'EST' 같은 약어는 지원되지 않을 수 있습니다. 