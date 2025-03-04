from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobotManualQueryHandler:
    def __init__(self):
        """Initialize the robot manual query handler."""
        self.common_topics = {
            "safety": ["safety", "warning", "caution", "emergency", "hazard", "protection"],
            "maintenance": ["maintenance", "repair", "service", "clean", "inspect", "check"],
            "operation": ["operation", "control", "navigate", "drive", "operate", "start", "stop"],
            "installation": ["installation", "setup", "configure", "mount", "install", "assembly"],
            "troubleshooting": ["troubleshoot", "error", "problem", "fault", "issue", "diagnostic"],
            "specifications": ["specification", "dimension", "weight", "capacity", "power", "battery"]
        }
        
    def categorize_query(self, query: str) -> List[str]:
        """Categorize the query into relevant topics.
        
        Args:
            query (str): User's query
            
        Returns:
            List[str]: List of relevant topics
        """
        query = query.lower()
        relevant_topics = []
        
        for topic, keywords in self.common_topics.items():
            if any(keyword in query for keyword in keywords):
                relevant_topics.append(topic)
                
        return relevant_topics
    
    def generate_focused_query(self, query: str) -> str:
        """Enhance the query with robot-specific context.
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Enhanced query
        """
        topics = self.categorize_query(query)
        
        # Add context based on identified topics
        if topics:
            context_additions = []
            for topic in topics:
                if topic == "safety":
                    context_additions.append("safety requirements and procedures")
                elif topic == "maintenance":
                    context_additions.append("maintenance procedures and schedules")
                elif topic == "operation":
                    context_additions.append("operational instructions and controls")
                elif topic == "installation":
                    context_additions.append("installation and setup procedures")
                elif topic == "troubleshooting":
                    context_additions.append("troubleshooting steps and solutions")
                elif topic == "specifications":
                    context_additions.append("technical specifications and requirements")
            
            enhanced_query = f"{query} in context of {', '.join(context_additions)}"
            return enhanced_query
        
        return query
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format search results with robot manual-specific structure.
        
        Args:
            results: Raw search results
            
        Returns:
            Dict[str, Any]: Formatted results with sections
        """
        formatted_results = {
            "main_answer": [],
            "related_sections": [],
            "manual_references": set(),
            "safety_notes": []
        }
        
        for result in results:
            # Extract manual name without extension
            manual_name = result['document_name'].replace('.pdf', '')
            formatted_results["manual_references"].add(manual_name)
            
            # Check if this is a safety-related result
            if any(word in result['document'].lower() for word in self.common_topics["safety"]):
                formatted_results["safety_notes"].append({
                    "text": result['document'],
                    "source": manual_name,
                    "relevance": result['score']
                })
            else:
                formatted_results["main_answer"].append({
                    "text": result['document'],
                    "source": manual_name,
                    "relevance": result['score']
                })
        
        # Convert manual_references to list for JSON serialization
        formatted_results["manual_references"] = list(formatted_results["manual_references"])
        
        return formatted_results 