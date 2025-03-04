import os
from typing import Dict, Any, List
import jinja2
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LatexReportGenerator:
    def __init__(self, template_dir: str = "templates"):
        """Initialize LaTeX report generator.
        
        Args:
            template_dir (str): Directory containing LaTeX templates
        """
        self.template_dir = template_dir
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            block_start_string='\\BLOCK{',
            block_end_string='}',
            variable_start_string='\\VAR{',
            variable_end_string='}',
            comment_start_string='\\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            autoescape=False
        )
        
    def generate_report(
        self,
        results: Dict[str, Any],
        output_dir: str,
        report_name: str = "model_evaluation_report"
    ) -> str:
        """Generate a LaTeX report from results.
        
        Args:
            results: Analysis results dictionary
            output_dir: Directory to save the report
            report_name: Name of the report file
            
        Returns:
            str: Path to generated PDF
        """
        # Prepare data for the template
        template_data = self._prepare_template_data(results)
        
        # Generate LaTeX content
        template = self.env.get_template("report_template.tex")
        latex_content = template.render(**template_data)
        
        # Save LaTeX file
        os.makedirs(output_dir, exist_ok=True)
        tex_path = os.path.join(output_dir, f"{report_name}.tex")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        # Compile PDF
        self._compile_pdf(tex_path)
        
        return tex_path.replace('.tex', '.pdf')
    
    def _prepare_template_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for the LaTeX template."""
        return {
            'title': 'Model Evaluation Report',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'model_performance': results['summary']['best_model_performance'],
            'cv_results': results['summary']['cross_validation_stability'],
            'optimization': results.get('summary', {}).get('optimization', {}),
            'figures': results['figures'],
            'metrics': results.get('metrics', {}),
            'timestamp': results.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        }
    
    def _compile_pdf(self, tex_path: str) -> None:
        """Compile LaTeX to PDF."""
        os.system(f"pdflatex -interaction=nonstopmode -output-directory={os.path.dirname(tex_path)} {tex_path}")
        # Run twice for references
        os.system(f"pdflatex -interaction=nonstopmode -output-directory={os.path.dirname(tex_path)} {tex_path}")
        
        # Clean up auxiliary files
        for ext in ['.aux', '.log', '.out']:
            aux_file = tex_path.replace('.tex', ext)
            if os.path.exists(aux_file):
                os.remove(aux_file) 