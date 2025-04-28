# Instruction Data Generator

## Configuration

The system uses a split configuration approach for better organization and flexibility:

### Configuration Files

1. `config/agent_config.yaml`: Contains settings for the agent's behavior, including:
   - Input processing parameters
   - Instruction generation settings
   - Quality control thresholds
   - Document processing configuration
   - Output format preferences

2. `config/model_config.yaml`: Contains all model-related configurations:
   - LLM model definitions and parameters
   - Model serving configurations (vLLM, Transformers, etc.)
   - GPU and resource allocation settings
   - Model-specific prompts and templates

3. `config/.env`: Environment-specific settings and secrets:
   - API keys
   - Model endpoints
   - Resource limits
   - Custom paths
   - A template is provided as `.env.template`

### Usage

1. Basic usage:
   ```bash
   python main.py
   ```
   This will use default config paths: `config/agent_config.yaml` and `config/model_config.yaml`

2. Custom config paths:
   ```bash
   python main.py --agent-config path/to/agent_config.yaml --model-config path/to/model_config.yaml
   ```

3. Specify model:
   ```bash
   python main.py --model llama2_70b
   ```
   Available models: llama2_70b, llama3_70b, qwen25_72b, qwen2_70b, llama2_13b, qwen_14b, phi35

4. Control resource usage:
   ```bash
   python main.py --max-memory 35GiB
   ```

### Environment Setup

1. Copy the environment template:
   ```bash
   cp config/.env.template config/.env
   ```

2. Edit `config/.env` with your settings:
   - Add your API keys
   - Set resource limits
   - Configure custom paths
   - Adjust model settings