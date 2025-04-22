# Instruction Data Generator

An AI agent for generating high-quality instruction data from multimodal sources for training large language models. This tool uses state-of-the-art models and techniques from 2025 to generate question-answer pairs from text and images while ensuring the generated data is derived solely from the input sources.

## Features

- Multimodal input support (text and images)
- Advanced image captioning using BLIP-2 or Phi-3.5-vision-instruct
- Sophisticated answer extraction and question generation
- Self-instruction techniques for data augmentation
- Comprehensive quality control and validation
- Support for multiple output formats
- Detailed logging and statistics

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-compatible GPU recommended for optimal performance

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/instruction-data-generator.git
cd instruction-data-generator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Download required model weights and resources:
```bash
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt
```

## Configuration

The agent uses a flexible configuration system with three main components:

### 1. Agent Configuration (`config/agent_config.yaml`)
Contains general agent settings including:
- Input/output processing
- Instruction generation parameters
- Quality control settings
- Document processing configuration

### 2. Model Configuration (`config/model_config.yaml`)
Contains model-specific settings:
- Model selection and parameters
- Optimization settings
- Default parameters
- Prompt templates

### 3. Environment Configuration (`config/.env`)
Contains sensitive and environment-specific values:
```env
# API Keys
OPENAI_API_KEY=your_api_key_here  # Required for GPT-4o
HF_TOKEN=your_huggingface_token   # Required for Hugging Face models

# Model Endpoints
PHI_MODEL_ENDPOINT=your_endpoint   # Optional custom endpoint

# Resource Limits
MAX_GPU_MEMORY=12GB
MAX_CPU_MEMORY=32GB
NUM_WORKERS=4

# Paths
CACHE_DIR=.cache
MODEL_DIR=models
OUTPUT_DIR=data/output
```

The configuration files are automatically merged at runtime, with environment variables taking precedence over YAML configurations.

## Usage

### Basic Usage

```python
from src.agent import InstructionDataGenerator

# Initialize the agent
agent = InstructionDataGenerator('config/agent_config.yaml')

# Generate instruction data
agent.generate_instruction_data(
    input_dir='data/input',
    output_dir='data/output'
)
```

### Command Line Interface

```bash
# Generate instructions from input data
generate-instructions --input data/input --output data/output
```

### Input Data Structure

Place your input data in the following structure:

```
data/
  input/
    text/
      document1.txt
      document2.md
      ...
    images/
      image1.jpg
      image2.png
      ...
```

### Output Format

The generator produces instruction data in JSONL format:

```jsonl
{"source": "text/document1.txt", "question": "...", "answer": "...", "metrics": {...}}
{"source": "images/image1.jpg", "question": "...", "answer": "...", "metrics": {...}}
```

## Quality Control

The agent implements multiple quality control measures:

- Answer relevance validation
- Question quality assessment
- Diversity metrics
- Source fidelity checks

Configure quality thresholds in `agent_config.yaml`:

```yaml
quality_control:
  min_quality_score: 0.7
  metrics:
    - "relevance"
    - "answerability"
    - "specificity"
    - "diversity"
```

## Advanced Features

### Self-Instruction

The agent can use existing QA pairs to generate additional high-quality pairs:

```python
agent.self_instruct.generate_instructions(
    text=content,
    seed_qa_pairs=existing_pairs,
    num_pairs=5
)
```

### Data Augmentation

Generate variations of existing QA pairs:

```python
augmented_pairs = agent.self_instruct.augment_with_variations(qa_pairs)
```

### Statistics

Get detailed statistics about generated data:

```python
stats = agent.get_statistics(qa_pairs)
print(json.dumps(stats, indent=2))
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{instruction_data_generator,
  title = {Instruction Data Generator},
  author = {ThanhMV},
  year = {2025},
  url = {https://github.com/yourusername/instruction-data-generator}
}
```