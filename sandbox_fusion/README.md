# Sandbox Fusion Tool

This directory contains the sandbox code execution functionality for Search-o1-serper-sandbox.

## Overview

The sandbox tool allows the model to execute code during inference, providing a safe environment for running user-provided code snippets.

## Configuration

The sandbox tool is configured via `sandbox_config.json`:

```json
{
  "sandbox_fusion_url": "https://sd58dqj3dr43upa0div0g.apigateway-cn-beijing.volceapi.com/run_code",
  "token": "your_token_here",
  "api_key": "your_api_key_here",
  "memory_limit_mb": 2048,
  "default_timeout": 30,
  "default_language": "python"
}
```

## Usage

### In Search-o1

To enable code execution in Search-o1, use the `--enable_code_execution` flag:

```bash
python scripts/run_search_o1.py --dataset_name livecode --enable_code_execution [other args...]
```

### Code Execution Format (Open-AgentRL Style)

The system now uses OpenAI function calling format (Hermes style) for tool calls, consistent with Open-AgentRL:

#### Tool Call Format:
```xml
<tool_call>
{
  "name": "code_interpreter",
  "arguments": {
    "code": "print('Hello from sandbox!')"
  }
}
</tool_call>
```

#### Tool Response Format:
```
<|begin_code_result|>
Hello from sandbox!
<|end_code_result|>
```

### Available Prompts

The system provides specialized prompts for different types of problems:

- `get_code_execution_instruction()`: General code execution instruction
- `get_math_code_execution_instruction()`: Math problems with code execution

### Example Usage

For a math problem, the model might generate:
```
Let me calculate 123^2 + 456^2 - 789^2.

<tool_call>
{
  "name": "code_interpreter",
  "arguments": {
    "code": "print(123**2 + 456**2 - 789**2)"
  }
}
</tool_call>
```

The system will execute the code and return:
```
<|begin_code_result|>
-234258
<|end_code_result|>
```

## Supported Languages

The sandbox supports multiple programming languages including:
- Python
- C++
- Java
- JavaScript (Node.js)
- Go
- And many more...

## Testing

Run the test script to verify sandbox functionality:

```bash
cd sandbox_fusion
python test_sandbox.py
```

## Dependencies

- requests
- json
- Other standard library modules

Make sure the sandbox API endpoint is accessible and properly configured.
