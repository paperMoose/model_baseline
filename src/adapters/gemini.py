from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from google import genai
from google.genai import types
from typing import List, Optional
from datetime import datetime, timezone
from src.schemas import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the Gemini model using genai.Client as per new SDK docs.
        """
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.generation_config_dict = self.model_config.kwargs # Store the kwargs
        
        # Initialize the client using genai.Client
        client = genai.Client(api_key=api_key)
        return client

    def make_prediction(self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None, pair_index: int = None) -> Attempt:
        """
        Make a prediction with the Gemini model and return an Attempt object
        
        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
        """
        start_time = datetime.now(timezone.utc)
        
        # For a single prompt, typically a user message
        messages = [{"role": "user", "content": prompt}] 
        response = self.chat_completion(messages)
        
        if response is None:
            logger.error(f"Failed to get response from chat_completion for task {task_id}")
            return Attempt(
                metadata=AttemptMetadata(
                    model=self.model_config.model_name,
                    provider=self.model_config.provider,
                    start_timestamp=start_time,
                    end_timestamp=datetime.now(timezone.utc),
                    choices=[], 
                    kwargs=self.model_config.kwargs, 
                    usage=Usage( # Provide default Usage object
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        completion_tokens_details=CompletionTokensDetails(
                            reasoning_tokens=0,
                            accepted_prediction_tokens=0,
                            rejected_prediction_tokens=0
                        )
                    ), 
                    cost=Cost( # Provide default Cost object
                        prompt_cost=0.0,
                        completion_cost=0.0,
                        total_cost=0.0
                    ),
                    error_message="Failed to get valid response from provider",
                    task_id=task_id, pair_index=pair_index, test_id=test_id # Ensure these are passed
                ),
                answer=""
            )

        end_time = datetime.now(timezone.utc)

        # Safely access usage_metadata and its attributes
        usage_metadata = getattr(response, 'usage_metadata', None)
        logger.debug(f"Response usage metadata: {usage_metadata}")
        
        input_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
        output_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
        total_tokens = getattr(usage_metadata, 'total_token_count', 0) if usage_metadata else 0
        
        response_text = getattr(response, 'text', "")

        # Use pricing from model config
        input_cost_per_token = self.model_config.pricing.input / 1_000_000
        output_cost_per_token = self.model_config.pricing.output / 1_000_000
        
        prompt_cost = input_tokens * input_cost_per_token
        completion_cost = output_tokens * output_cost_per_token

        input_choices = [
            Choice(index=i, message=Message(role=msg["role"], content=msg["content"]))
            for i, msg in enumerate(messages)
        ]
        response_choices = [
            Choice(index=len(input_choices), message=Message(role="assistant", content=response_text))
        ]
        all_choices = input_choices + response_choices

        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,
                    accepted_prediction_tokens=output_tokens,
                    rejected_prediction_tokens=0
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            ),
            task_id=task_id, pair_index=pair_index, test_id=test_id
        )
        attempt = Attempt(metadata=metadata, answer=response_text)
        return attempt

    def chat_completion(self, messages: list):
        contents_list = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                role = "model"  # Gemini uses 'model' for assistant
            
            # Ensure role is either 'user' or 'model' for multi-turn
            # System instructions are handled by GenerateContentConfig
            if role in ["user", "model"]:
                contents_list.append(types.Content(role=role, parts=[types.Part(text=content)]))
            elif role == "system" and content: # Handle system message if provided
                # If system_instruction is also in generation_config_dict, it might conflict or be preferred.
                # The API might prefer one over the other or merge. For now, include if present.
                # This assumes system messages can be part of the 'contents' list if structured correctly,
                # OR they are handled by system_instruction in GenerateContentConfig.
                # The docs imply system_instruction is part of GenerateContentConfig.
                # Let's ensure 'system_instruction' from kwargs takes precedence if it exists.
                if 'system_instruction' not in self.generation_config_dict:
                     # If not in main config, create a system instruction part if this is the only one
                     # This is a bit ambiguous from docs if contents can have system role directly.
                     # Sticking to system_instruction in GenerateContentConfig is safer.
                     # For now, let's assume system messages in the list are converted to user messages
                     # or handled by a top-level system_instruction.
                     # The most robust is to rely on system_instruction in GenerateContentConfig.
                     # We'll filter system messages out of contents_list and rely on the config.
                     logger.info(f"System message in chat history: '{content}'. Will be handled by system_instruction in GenerateContentConfig if provided.")
                pass # System messages from history are not added to contents_list directly
                     # They should be specified via `system_instruction` in `GenerateContentConfig`.

        # generation_config_dict already contains all kwargs including potential system_instruction
        # types.GenerateContentConfig will pick up system_instruction if it's a valid param for it.
        config_params = self.generation_config_dict.copy()

        try:
            response = self.client.models.generate_content(
                model=self.model_config.model_name, # Pass model name here
                contents=contents_list,
                config=types.GenerateContentConfig(**config_params)
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat_completion with google.genai (client.models.generate_content): {e}")
            if hasattr(e, 'response') and e.response:
                 logger.error(f"API Error details: {e.response}")
            return None

    def extract_json_from_response(self, input_response: str) -> Optional[List[List[int]]]:
        prompt = f"""
        Extract only the JSON of the test output from the following response.
        Remove any markdown code blocks and return only valid JSON.

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
        
        # Config for extraction should be minimal, e.g., temperature.
        # Filter from self.generation_config_dict
        extract_config_params = {
            k: v for k, v in self.generation_config_dict.items() 
            if k in ['temperature', 'top_p', 'top_k', 'max_output_tokens', 'stop_sequences'] # Common params for generation
        }
        # system_instruction is unlikely needed for simple extraction from existing text.

        try:
            response = self.client.models.generate_content(
                model=self.model_config.model_name, # Specify model
                contents=prompt, 
                config=types.GenerateContentConfig(**extract_config_params) if extract_config_params else None
            )
            content = response.text.strip()

            if content.startswith("```json"):
                content = content[7:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            try:
                json_data = json.loads(content)
                return json_data.get("response")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from extraction response: {content}")
                return None
        except Exception as e:
            logger.error(f"Error in extract_json_from_response with google.genai: {e}")
            return None