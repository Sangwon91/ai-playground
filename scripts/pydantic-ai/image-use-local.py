from pydantic_ai import Agent, BinaryContent
import dotenv
import logfire
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Literal
dotenv.load_dotenv()
logfire.configure(send_to_logfire=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_pydantic_ai()

image_path = Path(__file__).parent / 'assets' / '8-09.jpg'


class PointPrompt(BaseModel):
    x: float
    y: float
    label: Literal['pos', 'neg']

agent = Agent(
    model='google-gla:gemini-2.5-flash',
    system_prompt=(
        '당신은 이미지 세그멘테이션을 위한 오토 라벨링을 수행하는 모델입니다.'
        'Segment Anything 모델을 사용하여 라벨링을 수행하기 때문에 당신이 할 일은'
        '이미지 라벨링에 적절한 Segment Anything 모델의 프롬프트를 생성하는 것입니다.'
        '지금은 포인트 프롬프팅만 사용할 것이며 포인트 프롬프팅을 위한 프롬프트를 생성해주세요.'
        '좌표계는 이미지의 왼쪽 위가 0, 0이며 오른쪽 아래가 1, 1이며 좌표는 0.0 부터 1.0 까지 입니다.'
        'Positive, Negative 좌표를 모두 사용할 수 있습니다.'
        '좌표는 소수점 첫째자리까지만 출력해주세요.'
        '영역이 충분히 정확하게 포함될 수 있는 수의 포인트를 사용해주세요.'
    ),
    output_type=list[PointPrompt],
)
result = agent.run_sync(
    [
        '이 이미지에서 가장 큰 물체를 찾기 위한 포인트 프롬프팅을 생성해주세요.',
        BinaryContent(data=image_path.read_bytes(), media_type='image/jpeg'), 
    ]
)
print(result.output)
# > This is the logo for Pydantic, a data validation and settings management library in Python.