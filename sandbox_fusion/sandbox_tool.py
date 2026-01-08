import logging
import os
import json
from typing import Any, Optional, Dict

from .utils import call_sandbox_api

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))

class SandboxTool:
    """A simple sandbox tool for code execution."""

    def __init__(self, config: dict):
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        self.token = config.get("token", None)
        self.api_key = config.get("api_key", None)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")

        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")

        log_msg = f"Init SandboxTool with config: {config}"
        logger.info(log_msg)

    def execute_code(self, code: str, timeout: int = None, language: str = None) -> str:
        """Execute code in sandbox and return the result."""
        if timeout is None:
            timeout = self.default_timeout
        if language is None:
            language = self.default_language

        api_response, error_msg = call_sandbox_api(
            sandbox_fusion_url=self.sandbox_fusion_url,
            code=code,
            stdin=None,
            compile_timeout=timeout,
            run_timeout=timeout,
            memory_limit_mb=self.memory_limit_mb,
            language=language,
            token=self.token,
            api_key=self.api_key,
        )

        if error_msg:
            logger.error(f"Sandbox execution error: {error_msg}")
            return f"Error executing code: {error_msg}"

        if api_response:
            # Extract run result
            run_result = api_response.get("run_result")
            if run_result and run_result.get("status") == "Finished":
                stdout = run_result.get("stdout", "")
                stderr = run_result.get("stderr", "")
                return stdout + stderr
            else:
                return f"Sandbox execution failed with status: {run_result.get('status') if run_result else 'Unknown'}"
        else:
            return "No response from sandbox API"

def create_sandbox_tool(config_path: str = None) -> Optional[SandboxTool]:
    """Create a sandbox tool from configuration file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "sandbox_config.json")

    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return SandboxTool(config)
    except Exception as e:
        logger.error(f"Error loading sandbox configuration: {e}")
        return None
