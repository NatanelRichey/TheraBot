#!/usr/bin/env python3
"""
Data Cleaning and Labeling Script for TheraBot Training Data

This script automatically:
1. Standardizes speaker labels across all files
2. Infers therapy approaches from filenames and content
3. Extracts clinical topics using keyword matching
4. Detects session types from speaker patterns
5. Adds rich metadata for better training
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapyDataCleaner:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)
        self.speaker_mapping = {
            "THERAPIST": "therapist",
            "CLIENT": "client",
            "patient": "client",
            "FEMALE CLIENT": "client",
            "MALE CLIENT": "client",
            "COUNSELOR": "therapist",
            "PATIENT": "client",
            "DR. FELDER": "therapist",
            "DR. ELLIS": "therapist",
            "CARL ROGERS": "therapist",
            "WHITAKER": "therapist"
        }
        
        # Therapy approach keywords (filename patterns)
        self.approach_patterns = {
            "psychoanalytic": ["psychoanalytic", "psychodynamic", "freud", "unconscious"],
            "cbt": ["cbt", "cognitive", "behavioral", "beck", "ellis", "rebt"],
            "client_centered": ["client_centered", "person_centered", "rogers", "non_directive"],
            "integrative": ["integrative", "eclectic", "multi_modal"],
            "solution_focused": ["solution_focused", "brief_therapy", "sfbt"],
            "emdr": ["emdr", "trauma", "eye_movement"],
            "family_systems": ["family", "couples", "systemic", "whitaker"],
            "group": ["group_therapy", "group"],
            "dbt": ["dbt", "dialectical", "borderline"],
            "gestalt": ["gestalt", "perls"],
            "humanistic": ["humanistic", "existential", "maslow"]
        }
        
        # Clinical topic keywords
        self.topic_keywords = {
            "anxiety": ["anxiety", "anxious", "panic", "worry", "fear", "nervous", "apprehension"],
            "depression": ["depression", "depressed", "sad", "hopeless", "worthless", "suicidal"],
            "relationships": ["relationship", "marriage", "couple", "partner", "intimacy", "divorce"],
            "trauma": ["trauma", "ptsd", "abuse", "assault", "violence", "flashback"],
            "addiction": ["addiction", "alcohol", "drug", "substance", "recovery", "sober"],
            "grief": ["grief", "loss", "death", "mourning", "bereavement"],
            "work_stress": ["work", "job", "career", "boss", "colleague", "unemployment"],
            "sexual_dysfunction": ["sexual", "sex", "intimacy", "dysfunction", "arousal"],
            "self_esteem": ["self_esteem", "confidence", "worth", "value", "inadequate"],
            "anger": ["anger", "angry", "rage", "furious", "hostile", "irritated"],
            "eating_disorders": ["eating", "anorexia", "bulimia", "food", "weight", "body_image"],
            "personality": ["personality", "borderline", "narcissistic", "bpd", "npd"],
            "family_issues": ["family", "parent", "child", "sibling", "dysfunctional"],
            "life_transitions": ["transition", "change", "move", "graduation", "retirement"]
        }
        
        # Session type detection patterns
        self.session_type_patterns = {
            "individual": ["therapist", "client", "patient"],
            "couples": ["couple", "husband", "wife", "partner", "marriage"],
            "group": ["group", "members", "everyone", "others"],
            "family": ["family", "parent", "child", "sibling", "mother", "father"]
        }

    def detect_therapy_approach(self, filename: str, content: str) -> str:
        """Detect therapy approach from filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Check filename patterns first
        for approach, keywords in self.approach_patterns.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return approach
        
        # Check content for approach indicators
        for approach, keywords in self.approach_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return approach
        
        # Default based on collection name
        if "ellis" in filename_lower or "rebt" in filename_lower:
            return "cbt"
        elif "rogers" in filename_lower or "client_centered" in filename_lower:
            return "client_centered"
        elif "psychoanalytic" in filename_lower:
            return "psychoanalytic"
        elif "couples" in filename_lower or "couple" in filename_lower:
            return "family_systems"
        elif "group" in filename_lower:
            return "group"
        
        return "unknown"

    def detect_clinical_topics(self, content: str) -> List[str]:
        """Detect clinical topics from content using keyword matching"""
        content_lower = content.lower()
        detected_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    detected_topics.append(topic)
                    break  # Only add topic once
        
        return detected_topics

    def detect_session_type(self, speakers: List[str], content: str) -> str:
        """Detect session type from speaker patterns and content"""
        content_lower = content.lower()
        speakers_lower = [s.lower() for s in speakers]
        
        # Check for couples therapy
        if any(term in content_lower for term in ["couple", "husband", "wife", "partner", "marriage"]):
            return "couples"
        
        # Check for group therapy
        if any(term in content_lower for term in ["group", "members", "everyone", "others"]):
            return "group"
        
        # Check for family therapy
        if any(term in content_lower for term in ["family", "parent", "child", "sibling", "mother", "father"]):
            return "family"
        
        # Check speaker patterns
        if "female client" in speakers_lower and "male client" in speakers_lower:
            return "couples"
        
        # Default to individual
        return "individual"

    def detect_client_demographics(self, content: str, filename: str) -> Dict[str, str]:
        """Detect client demographics from content and filename"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        demographics = {}
        
        # Gender detection
        if any(term in content_lower for term in ["he said", "he feels", "he is", "his", "him"]):
            demographics["gender"] = "male"
        elif any(term in content_lower for term in ["she said", "she feels", "she is", "her", "hers"]):
            demographics["gender"] = "female"
        elif "male" in filename_lower:
            demographics["gender"] = "male"
        elif "female" in filename_lower:
            demographics["gender"] = "female"
        
        # Age detection (rough estimates)
        if any(term in content_lower for term in ["teenager", "teen", "high school", "college"]):
            demographics["age_range"] = "18-25"
        elif any(term in content_lower for term in ["graduate", "grad school", "young adult"]):
            demographics["age_range"] = "25-35"
        elif any(term in content_lower for term in ["middle aged", "midlife", "40s", "50s"]):
            demographics["age_range"] = "35-55"
        elif any(term in content_lower for term in ["elderly", "senior", "retirement"]):
            demographics["age_range"] = "55+"
        
        return demographics

    def standardize_speaker_labels(self, conversations: List[Dict]) -> List[Dict]:
        """Standardize speaker labels across all conversations"""
        standardized = []
        
        for conv in conversations:
            speaker = conv.get("speaker", "").strip()
            message = conv.get("message", conv.get("text", "")).strip()
            
            # Map speaker to standard format
            standardized_speaker = self.speaker_mapping.get(speaker, speaker.lower())
            
            standardized.append({
                "speaker": standardized_speaker,
                "message": message
            })
        
        return standardized

    def clean_and_enhance_file(self, file_path: Path) -> Dict[str, Any]:
        """Clean and enhance a single JSON file"""
        logger.info(f"Processing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
        
        # Extract content for analysis
        content = ""
        conversations = []
        
        # Handle different file structures
        if "dialogue" in data:
            conversations = data["dialogue"]
            content = " ".join([conv.get("text", conv.get("message", "")) for conv in conversations])
        elif "conversations" in data:
            conversations = data["conversations"]
            content = " ".join([conv.get("message", conv.get("text", "")) for conv in conversations])
        else:
            logger.warning(f"Unknown structure in {file_path}")
            return None
        
        # Standardize speaker labels
        standardized_conversations = self.standardize_speaker_labels(conversations)
        
        # Extract speakers for analysis
        speakers = list(set([conv["speaker"] for conv in standardized_conversations]))
        
        # Detect metadata
        therapy_approach = self.detect_therapy_approach(file_path.name, content)
        clinical_topics = self.detect_clinical_topics(content)
        session_type = self.detect_session_type(speakers, content)
        demographics = self.detect_client_demographics(content, file_path.name)
        
        # Count exchanges
        num_exchanges = len(standardized_conversations)
        
        # Create enhanced metadata
        enhanced_metadata = {
            "session_id": data.get("session_id", file_path.stem),
            "title": data.get("title", data.get("metadata", {}).get("title", "")),
            "therapy_approach": therapy_approach,
            "session_type": session_type,
            "clinical_topics": clinical_topics,
            "client_demographics": demographics,
            "num_exchanges": num_exchanges,
            "therapist": data.get("therapist", "Unknown"),
            "client_info": data.get("client_info", "Anonymous"),
            "original_format": data.get("metadata", {}).get("format", "unknown"),
            "file_source": str(file_path)
        }
        
        # Create enhanced data structure
        enhanced_data = {
            "metadata": enhanced_metadata,
            "conversations": standardized_conversations
        }
        
        return enhanced_data

    def process_all_files(self, output_dir: str = "data/processed_cleaned"):
        """Process all JSON files in the processed directory"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        processed_count = 0
        total_exchanges = 0
        approach_counts = {}
        topic_counts = {}
        
        # Find all JSON files (excluding summary files)
        json_files = list(self.processed_dir.rglob("*.json"))
        json_files = [f for f in json_files if "summary" not in f.name.lower()]
        
        logger.info(f"Found {len(json_files)} files to process")
        
        for file_path in json_files:
            enhanced_data = self.clean_and_enhance_file(file_path)
            
            if enhanced_data is None:
                continue
            
            # Save enhanced file
            relative_path = file_path.relative_to(self.processed_dir)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            processed_count += 1
            exchanges = enhanced_data["metadata"]["num_exchanges"]
            total_exchanges += exchanges
            
            approach = enhanced_data["metadata"]["therapy_approach"]
            approach_counts[approach] = approach_counts.get(approach, 0) + 1
            
            for topic in enhanced_data["metadata"]["clinical_topics"]:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            logger.info(f"Processed {file_path.name}: {exchanges} exchanges, {approach} approach")
        
        # Create summary report
        summary = {
            "processing_date": "2025-01-27",
            "total_files_processed": processed_count,
            "total_exchanges": total_exchanges,
            "therapy_approaches": approach_counts,
            "clinical_topics": topic_counts,
            "output_directory": str(output_path)
        }
        
        with open(output_path / "cleaning_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete!")
        logger.info(f"Files processed: {processed_count}")
        logger.info(f"Total exchanges: {total_exchanges:,}")
        logger.info(f"Therapy approaches: {approach_counts}")
        logger.info(f"Clinical topics: {topic_counts}")
        
        return summary

def main():
    """Main function to run the data cleaning process"""
    cleaner = TherapyDataCleaner()
    summary = cleaner.process_all_files()
    
    print("\n" + "="*50)
    print("DATA CLEANING COMPLETE")
    print("="*50)
    print(f"Files processed: {summary['total_files_processed']}")
    print(f"Total exchanges: {summary['total_exchanges']:,}")
    print(f"\nTherapy approaches found:")
    for approach, count in summary['therapy_approaches'].items():
        print(f"  {approach}: {count} files")
    print(f"\nClinical topics found:")
    for topic, count in summary['clinical_topics'].items():
        print(f"  {topic}: {count} files")
    print(f"\nCleaned data saved to: {summary['output_directory']}")

if __name__ == "__main__":
    main()
