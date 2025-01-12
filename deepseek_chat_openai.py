from typing import Type, Dict, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import get_type_hints
import json

class DeepSeekChatOpenAI(ChatOpenAI):
    """
    A custom ChatOpenAI class that uses DeepSeek's API endpoint.
    Inherits from ChatOpenAI and modifies the API base URL.
    """
    def __init__(self, *args, **kwargs):
        # Override the API base URL to use DeepSeek's endpoint
        kwargs['openai_api_base'] = 'https://api.deepseek.com'
        super().__init__(*args, **kwargs)
        
    def with_structured_output(self, model_class: Type[BaseModel]):
        """
        Creates a function that processes input text and returns structured output based on a Pydantic model.
        
        Args:
            model_class (Type[BaseModel]): The Pydantic model class that defines the structure of the output.
            
        Returns:
            function: A function that takes input text and returns a structured Pydantic object.
        """
        
        def get_model_properties(cls: Type[BaseModel]) -> str:
            """
            Extracts property descriptions from a Pydantic model class.
            
            Args:
                cls (Type[BaseModel]): The Pydantic model class to extract properties from.
                
            Returns:
                str: A string containing all property names and their descriptions, one per line.
            """
            properties = []
            for field_name, field in cls.model_fields.items():
                description = field.description or field_name
                properties.append(f"{field_name}: {description}")
            return '\n'.join(properties)
        
        def store_as_pydantic(content: Union[str, Dict], cls: Type[BaseModel]) -> BaseModel:
            """
            Converts JSON data (either as string or dict) into a Pydantic model instance.
            Handles nested Pydantic models recursively.
            
            Args:
                content (Union[str, Dict]): The JSON data to convert.
                cls (Type[BaseModel]): The target Pydantic model class.
                
            Returns:
                BaseModel: An instance of the specified Pydantic model.
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
                    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                        # Recursively handle nested Pydantic models
                        parsed_data[field_name] = store_as_pydantic(field_value, field_type)
                    else:
                        parsed_data[field_name] = field_value
            
            return cls(**parsed_data)
        
        def structured_output_chain(input_prompt):
            """
            Creates and executes a chain that processes input text and returns structured output.
            
            Args:
                input_prompt (str): The input text to process.
                
            Returns:
                BaseModel: A Pydantic model instance containing the structured output.
            """
            # Define the prompt template that instructs the model to output JSON
            template = ChatPromptTemplate.from_template(
                """
                Please output the following information in JSON format. Only extract these properties:

                {properties}

                Content:

                {input}

                Please strictly output the result in the JSON format specified above, 
                including only the fields mentioned and ensuring correct types. 
                No code block wrapping is needed.
                """
            )
            
            # Get the properties from the model class and create the prompt
            properties = get_model_properties(model_class)
            prompt = template.invoke({
                "properties": properties,
                "input": input_prompt
            })
            
            # Get the response and convert it to a Pydantic model instance
            response = self.invoke(prompt)
            response_json = json.loads(response.content)
            return store_as_pydantic(response_json, model_class)
            
        return structured_output_chain
