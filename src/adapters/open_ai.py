from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from datetime import datetime
from src.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt

load_dotenv()

class OpenAIAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the OpenAI client
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = OpenAI()
        return client

    def make_prediction(self, prompt: str) -> Attempt:
        """
        Make a prediction with the OpenAI model and return an Attempt object
        """
        start_time = datetime.utcnow()
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.chat_completion(messages)
        
        end_time = datetime.utcnow()

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000  # Convert from per 1M tokens
        output_cost_per_token = self.model_config.pricing.output / 1_000_000  # Convert from per 1M tokens
        
        prompt_cost = response.usage.prompt_tokens * input_cost_per_token
        completion_cost = response.usage.completion_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=i,
                message=Message(
                    role=msg["role"],
                    content=msg["content"]
                )
            )
            for i, msg in enumerate(messages)
        ]

        # Convert OpenAI response to our schema
        response_choices = [
            Choice(
                index=len(input_choices),
                message=Message(
                    role=response.choices[0].message.role,
                    content=response.choices[0].message.content
                )
            )
        ]

        # Combine input and response choices
        all_choices = input_choices + response_choices

        # Create metadata using our Pydantic models
        metadata = AttemptMetadata(
            model=self.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # OpenAI doesn't provide this breakdown
                    accepted_prediction_tokens=response.usage.completion_tokens,
                    rejected_prediction_tokens=0  # OpenAI doesn't provide this
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            )
        )

        attempt = Attempt(
            metadata=metadata,
            answer=response.choices[0].message.content.strip()
        )

        return attempt

    def chat_completion(self, messages: list) -> str:
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self.model_config.kwargs
        )

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        prompt = f"""
You are a helpful assistant. Extract only the JSON of the test output from the following response. 
Do not include any explanation or additional text; only return valid JSON.

Response:
{input_response}

The JSON should be in this format:
{{
"response": [
    [1, 2, 3],
    [4, 5, 6]
]
}}
"""
        completion = self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )

        assistant_content = completion.choices[0].message.content.strip()

        # Some models like to wrap the response in a code block
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])

        try:
            json_entities = json.loads(assistant_content)
            return json_entities.get("response")
        except json.JSONDecodeError:
            return None