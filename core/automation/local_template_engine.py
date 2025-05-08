"""
LocalTemplateEngine: Jinja2-based template engine for zero-API-cost, high-performance offer, email, and campaign asset generation.
- Integrates with MonetizationEngine and Orchestrator for adaptive, hyper-personalized content.
- Supports dynamic context, segment, and trend adaptation.
"""
from jinja2 import Environment, FileSystemLoader, Template
import os
from typing import Dict, Any

class LocalTemplateEngine:
    def __init__(self, template_dir: str = "templates"):
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template_dir = template_dir

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        return template.render(**context)

    def register_template(self, template_name: str, template_str: str):
        path = os.path.join(self.template_dir, template_name)
        with open(path, "w") as f:
            f.write(template_str)
        self.env.loader = FileSystemLoader(self.template_dir)

    def list_templates(self):
        return os.listdir(self.template_dir)
