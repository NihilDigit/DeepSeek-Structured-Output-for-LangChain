from typing import Type, Dict, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import get_type_hints
import json

class DeepSeekChatOpenAI(ChatOpenAI):
    """
    Custom ChatOpenAI class that uses the DeepSeek API endpoint.
    Inherits from ChatOpenAI and extends it with structured output capabilities.
    """
    def __init__(self, *args, **kwargs):
        # Override the API base URL to use DeepSeek's endpoint
        kwargs['openai_api_base'] = 'https://api.deepseek.com'
        super().__init__(*args, **kwargs)
        
    def with_structured_output(self, model_class: Type[BaseModel]):
        """
        Creates a function that processes input text and returns structured output based on a Pydantic model.
        
        Args:
            model_class: A Pydantic BaseModel class that defines the structure of the output
            
        Returns:
            A function that takes input text and returns a structured Pydantic object
        """
        
        def get_model_properties(cls: Type[BaseModel]) -> str:
            """
            Extracts property descriptions from a Pydantic model class.
            
            Args:
                cls: The Pydantic model class to analyze
                
            Returns:
                A string containing all field names and their descriptions, with enum constraints if any
            """
            properties = []
            for field_name, field in cls.model_fields.items():
                description = field.description or field_name
                
                # Add enum constraints to the description if they exist
                if field.json_schema_extra and 'enum' in field.json_schema_extra:
                    enum_values = field.json_schema_extra['enum']
                    description = f"{description} (must be one of: {enum_values})"
                
                properties.append(f"{field_name}: {description}")
            return '\n'.join(properties)
        
        def store_as_pydantic(content: Union[str, Dict], cls: Type[BaseModel]) -> BaseModel:
            """
            Converts JSON data or dictionary into a Pydantic model instance.
            Handles nested Pydantic models recursively.
            
            Args:
                content: JSON string or dictionary containing the data
                cls: The target Pydantic model class
                
            Returns:
                An instance of the specified Pydantic model
            """
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
                
            field_types = get_type_hints(cls)
            parsed_data = {}
            
            # Process each field according to its type
            for field_name, field_type in field_types.items():
                field_value = data.get(field_name)
                if field_value is not None:
                    # Recursively handle nested Pydantic models
                    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                        parsed_data[field_name] = store_as_pydantic(field_value, field_type)
                    else:
                        parsed_data[field_name] = field_value
            
            return cls(**parsed_data)
        
        def structured_output_chain(input_prompt):
            """
            Creates and executes a chain that processes input text and returns structured output.
            
            Args:
                input_prompt: The input text to be processed
                
            Returns:
                A Pydantic model instance containing the structured output
            """
            template = ChatPromptTemplate.from_template(
                """
                Please output the following information in JSON format. Only extract the following properties, noting the value constraints for each field:

                {properties}

                Content:

                {input}

                Please strictly follow the JSON format above, include only the specified fields with correct types, and ensure enum field values are within the allowed range. No code block wrapping needed.
                """
            )
            
            # Get model properties and create the prompt
            properties = get_model_properties(model_class)
            prompt = template.invoke({
                "properties": properties,
                "input": input_prompt
            })
            
            # Get the response and convert it to a Pydantic model
            response = self.invoke(prompt)
            response_json = json.loads(response.content)
            return store_as_pydantic(response_json, model_class)
            
        return structured_output_chain
