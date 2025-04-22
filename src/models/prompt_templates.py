from typing import Dict, List, Any
from langchain.prompts import PromptTemplate
import yaml

class PromptTemplates:
    """Manages prompt templates for different instruction generation tasks."""
    
    def __init__(self):
        """Initialize prompt templates."""
        self.templates = self._load_default_templates()
        
    def _load_default_templates(self) -> Dict[str, Any]:
        """Load default prompt templates."""
        return {
            "image_caption": {
                "template": "Describe this image in detail, focusing only on what is visually present. Include all visible elements, objects, people, and text.",
                "input_variables": ["image"]
            },
            "question_generation": {
                "template": "Generate a question that can only be answered using the following text. The question should be clear, specific, and directly related to the given content:\n\nContext: {context}\nAnswer: {answer}\n\nGenerate a question for this answer:",
                "input_variables": ["context", "answer"]
            },
            "question_generation_factual": {
                "template": "Based on the following fact, generate a question that tests understanding of this specific information. The question must be answerable solely from this fact:\n\nFact: {fact}\n\nGenerate a clear and specific question:",
                "input_variables": ["fact"]
            },
            "self_instruct": {
                "template": "Given the following text, generate additional question-answer pairs that can be answered using ONLY the information provided in the text. Do not include external knowledge.\n\nText: {text}\n\nExisting Q&A pairs for reference:\n{examples}\n\nGenerate {num_pairs} new question-answer pairs:",
                "input_variables": ["text", "examples", "num_pairs"]
            },
            "answer_validation": {
                "template": "Verify if the answer can be fully derived from the given context. Consider only the information present in the context.\n\nContext: {context}\nQuestion: {question}\nAnswer: {answer}\n\nIs this answer fully supported by the context? Respond with 'Valid' or 'Invalid' and explain why:",
                "input_variables": ["context", "question", "answer"]
            }
        }
    
    def get_prompt(self, task_type: str, **kwargs) -> str:
        """
        Get a formatted prompt for a specific task.
        
        Args:
            task_type: Type of prompt template to use
            **kwargs: Variables to format the template with
            
        Returns:
            Formatted prompt string
        """
        if task_type not in self.templates:
            raise ValueError(f"No template found for task type: {task_type}")
            
        template_data = self.templates[task_type]
        prompt = PromptTemplate(
            template=template_data["template"],
            input_variables=template_data["input_variables"]
        )
        
        return prompt.format(**kwargs)
    
    def add_template(self, task_type: str, template: str, input_variables: List[str]) -> None:
        """
        Add a new prompt template.
        
        Args:
            task_type: Type of the new template
            template: The prompt template string
            input_variables: List of variables used in the template
        """
        self.templates[task_type] = {
            "template": template,
            "input_variables": input_variables
        }
    
    def save_templates(self, file_path: str) -> None:
        """
        Save current templates to a YAML file.
        
        Args:
            file_path: Path to save the templates
        """
        with open(file_path, 'w') as f:
            yaml.dump(self.templates, f)
    
    def load_templates(self, file_path: str) -> None:
        """
        Load templates from a YAML file.
        
        Args:
            file_path: Path to the templates file
        """
        with open(file_path, 'r') as f:
            self.templates = yaml.safe_load(f)