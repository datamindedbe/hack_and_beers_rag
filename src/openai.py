
import openai
from typing import Tuple
import time


def get_response(client:openai.OpenAI, prompt:str, model)->Tuple[str, float]:
    start_time = time.time()
    api_response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    exec_time = time.time() - start_time
    return (api_response.choices[0].message.content, exec_time)