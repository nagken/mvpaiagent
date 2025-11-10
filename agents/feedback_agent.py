"""
Feedback Agent - Handles logging and feedback collection
"""

import json
import os
import datetime
from typing import Dict, Any


class FeedbackAgent:
    """
    Collects and logs feedback from the classification pipeline
    for continuous learning and monitoring.
    """
    
    def __init__(self, log_file: str = "logs/agent_logs.jsonl"):
        self.log_file = log_file
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Create logs directory if it doesn't exist."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def run(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log classification result and return feedback summary.
        
        Args:
            result: Classification and validation result
            
        Returns:
            Feedback summary with logging status
        """
        try:
            # Prepare log entry
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self._generate_session_id(),
                "classification_result": result,
                "system_info": {
                    "agent_version": "1.0.0",
                    "pipeline_stage": "complete"
                }
            }
            
            # Write to log file
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            feedback_summary = {
                "status": "logged",
                "timestamp": log_entry["timestamp"],
                "session_id": log_entry["session_id"],
                "log_file": self.log_file
            }
            
            print(f"Logged classification result to {self.log_file}")
            
            return feedback_summary
            
        except Exception as e:
            print(f"Error logging feedback: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def log_user_feedback(self, session_id: str, user_rating: int, user_comments: str = "") -> Dict[str, Any]:
        """
        Log user feedback for a specific classification session.
        
        Args:
            session_id: Session ID from original classification
            user_rating: User rating (1-5)
            user_comments: Optional user comments
            
        Returns:
            Feedback logging result
        """
        try:
            feedback_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "feedback_type": "user_rating",
                "user_rating": user_rating,
                "user_comments": user_comments,
                "system_info": {
                    "feedback_version": "1.0.0"
                }
            }
            
            # Append to same log file with feedback prefix
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
            
            print(f"User feedback logged for session {session_id}")
            
            return {
                "status": "feedback_logged",
                "session_id": session_id,
                "timestamp": feedback_entry["timestamp"]
            }
            
        except Exception as e:
            print(f"Error logging user feedback: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_logs_summary(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent logs summary.
        
        Args:
            limit: Number of recent entries to analyze
            
        Returns:
            Summary statistics from recent logs
        """
        try:
            if not os.path.exists(self.log_file):
                return {"message": "No logs found", "total_entries": 0}
            
            # Read recent log entries
            entries = []
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            # Get most recent entries
            recent_entries = entries[-limit:] if len(entries) > limit else entries
            
            # Calculate summary stats
            total_classifications = len([e for e in recent_entries if "classification_result" in e])
            total_feedback = len([e for e in recent_entries if e.get("feedback_type") == "user_rating"])
            
            # Calculate average confidence
            confidences = []
            for entry in recent_entries:
                if "classification_result" in entry:
                    result = entry["classification_result"]
                    if "validated_prediction" in result:
                        conf = result["validated_prediction"].get("confidence", 0)
                        confidences.append(conf)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "total_entries": len(entries),
                "recent_classifications": total_classifications,
                "recent_feedback": total_feedback,
                "average_confidence": round(avg_confidence, 3),
                "log_file": self.log_file,
                "analysis_limit": limit
            }
            
        except Exception as e:
            print(f"Error analyzing logs: {e}")
            return {
                "error": str(e),
                "log_file": self.log_file
            }
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}_{os.urandom(4).hex()}"