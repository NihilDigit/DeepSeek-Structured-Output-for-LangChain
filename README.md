# Problem Solved
When using DeepSeek models with LangChain's ChatOpenAI, the built-in with_structured_output() method is not supported. This implementation provides a solution by extending the ChatOpenAI class with DeepSeek-compatible structured output functionality.

# Features
- Custom DeepSeekChatOpenAI class that inherits from LangChain's ChatOpenAI
- Automatic API endpoint configuration for DeepSeek
- Support for nested Pydantic models
- Type-safe JSON parsing
- Customizable output structure using Pydantic models

# Usage
```python
from deepseek_chat_openai.py import DeepSeekChatOpenAI
from pydantic import BaseModel

# Define your output structure using Pydantic
class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Initialize the model
model = DeepSeekChatOpenAI(
    api_key="your-api-key",
    model_name="deepseek-chat"
)

# Create a structured output chain
chain = model.with_structured_output(Person)

# Use the chain
result = chain("John is a 30-year-old software engineer.")

print(result.dict()) # {'name': 'John', 'age': 30, 'occupation': 'software engineer'}
print(result.name) # John
print(result.age + 1) # 31

```
