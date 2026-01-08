#!/usr/bin/env python3
"""
Example usage of the code execution functionality in Search-o1-serper-sandbox.

This example shows how to use the Open-AgentRL style tool calling format
for code execution in the Search-o1 system.
"""

# Example prompt for a math problem with code execution
MATH_PROBLEM_PROMPT = """
Analyze and solve the following math problem step by step.

The tool could be used for more precise and efficient calculation and could help you to verify your result before you reach the final answer.

**Available Tool:**
- `code_interpreter`: Execute Python code and return the output.
  Parameters:
  - `code` (string): The Python code to execute.

**How to use tools:**
To call a tool, wrap your tool call in <tool_call> and </tool_call> tags. The tool call should be a valid JSON object with 'name' and 'arguments' fields.

Example:
<tool_call>
{
  "name": "code_interpreter",
  "arguments": {
    "code": "print(2 + 3)"
  }
}
</tool_call>

**Note:** You should first analyze the problem and form a high-level solution strategy, then utilize the tools to help you solve the problem.

Remember once you make sure the current answer is your final answer, do not call the tools again and directly output the final answer in the following text format, the answer format must be: \\boxed{'The final answer goes here.'}.

Question: Calculate the value of 123^2 + 456^2 - 789^2.
"""

# Example prompt for a coding problem
CODING_PROBLEM_PROMPT = """
You are a reasoning assistant with access to a code execution tool. You can execute Python code to test your solutions and verify correctness.

**Available Tool:**
- `code_interpreter`: Execute Python code and return the output.
  Parameters:
  - `code` (string): The Python code to execute.

**How to use tools:**
To call a tool, wrap your tool call in <tool_call> and </tool_call> tags. The tool call should be a valid JSON object with 'name' and 'arguments' fields.

Example:
<tool_call>
{
  "name": "code_interpreter",
  "arguments": {
    "code": "print('Hello, World!')"
  }
}
</tool_call>

**Note:** You should first analyze the problem and form a high-level solution strategy, then utilize the tools to help you solve the problem. Before submitting your final code, you can utilize tools to check the correctness of your code. Once you make sure the current code is correct, do not call the tools again and directly submit your final code within ```python\n# YOUR CODE HERE\n```.

Problem: Write a Python function that takes a list of integers and returns the sum of all even numbers in the list.
"""

if __name__ == "__main__":
    print("Example usage of Search-o1-serper-sandbox with code execution:")
    print("=" * 60)
    print("Math Problem Example:")
    print(MATH_PROBLEM_PROMPT)
    print("\n" + "=" * 60)
    print("Coding Problem Example:")
    print(CODING_PROBLEM_PROMPT)
