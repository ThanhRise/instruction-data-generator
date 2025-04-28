import unittest
from pathlib import Path
import os
import tempfile
import yaml
from src.utils.helpers import (
    load_yaml_config,
    _validate_agent_config,
    _validate_model_config
)

class TestConfigLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up test configuration directory
        cls.test_dir = Path(__file__).parent.parent
        cls.test_config_path = cls.test_dir / "config" / "test_config.yaml"

    def test_load_yaml_config(self):
        """Test loading YAML configuration file"""
        config = load_yaml_config(self.test_config_path)
        
        # Verify basic structure
        self.assertIn("agent", config)
        self.assertIn("models", config)
        
        # Verify agent section
        agent_config = config["agent"]
        self.assertIn("input_processing", agent_config)
        self.assertIn("instruction_generation", agent_config)
        self.assertIn("quality_control", agent_config)
        
        # Verify models section
        models_config = config["models"]
        self.assertIn("llm_models", models_config)
        self.assertIn("serving", models_config)

    def test_validate_agent_config(self):
        """Test agent configuration validation"""
        with open(self.test_config_path) as f:
            config = yaml.safe_load(f)
        
        # Test valid config
        self.assertTrue(_validate_agent_config(config["agent"]))
        
        # Test invalid config
        invalid_config = {
            "input_processing": {},  # Missing required sections
        }
        self.assertFalse(_validate_agent_config(invalid_config))

    def test_validate_model_config(self):
        """Test model configuration validation"""
        with open(self.test_config_path) as f:
            config = yaml.safe_load(f)
        
        # Test valid config
        self.assertTrue(_validate_model_config(config["models"]))
        
        # Test invalid config
        invalid_config = {
            "llm_models": {
                "test_model": {
                    "name": "test"  # Missing required type field
                }
            }
        }
        self.assertFalse(_validate_model_config(invalid_config))

    def test_load_missing_file(self):
        """Test handling of missing configuration file"""
        with self.assertRaises(FileNotFoundError):
            load_yaml_config("nonexistent_config.yaml")

    def test_load_invalid_yaml(self):
        """Test handling of invalid YAML file"""
        # Create temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("invalid: yaml: content:")
        
        with self.assertRaises(yaml.YAMLError):
            load_yaml_config(temp_file.name)
        
        os.unlink(temp_file.name)

if __name__ == '__main__':
    unittest.main()