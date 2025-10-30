# Checkpoint Management Utilities
# For HuggingFace Hub and Google Drive integration
# Reference: https://huggingface.co/docs/hub/

import os
import json
import shutil
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import zipfile
from datetime import datetime

# Google Drive integration
try:
    from google.colab import drive
    from google.colab import files
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

# HuggingFace Hub integration
try:
    from huggingface_hub import HfApi, Repository, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Comprehensive checkpoint management for training.
    
    Handles saving, loading, and backing up model checkpoints
    to both HuggingFace Hub and Google Drive.
    """
    
    def __init__(self, 
                 hf_repo_name: str = "therabot-llama-3.1-8b-therapy-lora",
                 hf_private: bool = True,
                 google_drive_path: str = "/content/drive/MyDrive/TheraBot/checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            hf_repo_name: HuggingFace repository name
            hf_private: Whether to use private repository
            google_drive_path: Google Drive backup path
        """
        self.hf_repo_name = hf_repo_name
        self.hf_private = hf_private
        self.google_drive_path = google_drive_path
        self.hf_api = HfApi() if HF_HUB_AVAILABLE else None
        
        # Create backup directories
        self._setup_backup_directories()
    
    def _setup_backup_directories(self):
        """Setup backup directories for checkpoints."""
        # Create Google Drive backup directory
        if COLAB_AVAILABLE:
            try:
                drive.mount('/content/drive')
                os.makedirs(self.google_drive_path, exist_ok=True)
                logger.info(f"Google Drive backup directory created: {self.google_drive_path}")
            except Exception as e:
                logger.warning(f"Could not setup Google Drive backup: {e}")
        
        # Create local backup directory
        self.local_backup_path = "./checkpoint_backups"
        os.makedirs(self.local_backup_path, exist_ok=True)
        logger.info(f"Local backup directory created: {self.local_backup_path}")
    
    def create_hf_repository(self, token: str) -> bool:
        """
        Create HuggingFace repository if it doesn't exist.
        
        Args:
            token: HuggingFace authentication token
            
        Returns:
            True if repository created successfully
        """
        if not HF_HUB_AVAILABLE:
            logger.error("HuggingFace Hub not available")
            return False
        
        try:
            # Check if repository exists
            try:
                self.hf_api.repo_info(self.hf_repo_name)
                logger.info(f"Repository {self.hf_repo_name} already exists")
                return True
            except RepositoryNotFoundError:
                pass
            
            # Create new repository
            create_repo(
                repo_id=self.hf_repo_name,
                private=self.hf_private,
                token=token,
                repo_type="model"
            )
            logger.info(f"Created HuggingFace repository: {self.hf_repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create HuggingFace repository: {e}")
            return False
    
    def save_checkpoint(self, 
                       model, 
                       tokenizer, 
                       training_args,
                       run_name: str,
                       metrics: Optional[Dict] = None,
                       push_to_hub: bool = True) -> str:
        """
        Save model checkpoint with comprehensive backup.
        
        Args:
            model: The trained model
            tokenizer: The tokenizer
            training_args: Training arguments
            run_name: Name for this training run
            metrics: Optional training metrics
            push_to_hub: Whether to push to HuggingFace Hub
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{run_name}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = os.path.join(training_args.output_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save training arguments
        training_args.save_to_json(os.path.join(checkpoint_path, "training_args.json"))
        
        # Save metrics if provided
        if metrics:
            with open(os.path.join(checkpoint_path, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        
        # Save checkpoint metadata
        metadata = {
            "checkpoint_name": checkpoint_name,
            "run_name": run_name,
            "timestamp": timestamp,
            "model_name": getattr(model.config, "name_or_path", "unknown"),
            "training_args": training_args.to_dict(),
            "metrics": metrics or {}
        }
        
        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved locally: {checkpoint_path}")
        
        # Backup to Google Drive
        self._backup_to_google_drive(checkpoint_path, checkpoint_name)
        
        # Push to HuggingFace Hub if requested
        if push_to_hub and HF_HUB_AVAILABLE:
            self._push_to_huggingface_hub(checkpoint_path, checkpoint_name)
        
        return checkpoint_path
    
    def _backup_to_google_drive(self, checkpoint_path: str, checkpoint_name: str):
        """Backup checkpoint to Google Drive."""
        if not COLAB_AVAILABLE:
            logger.warning("Google Colab not available - skipping Drive backup")
            return
        
        try:
            # Create zip file for backup
            zip_path = os.path.join(self.local_backup_path, f"{checkpoint_name}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(checkpoint_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, checkpoint_path)
                        zipf.write(file_path, arc_path)
            
            # Copy to Google Drive
            drive_backup_path = os.path.join(self.google_drive_path, f"{checkpoint_name}.zip")
            shutil.copy2(zip_path, drive_backup_path)
            
            logger.info(f"Checkpoint backed up to Google Drive: {drive_backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to backup to Google Drive: {e}")
    
    def _push_to_huggingface_hub(self, checkpoint_path: str, checkpoint_name: str):
        """Push checkpoint to HuggingFace Hub."""
        if not HF_HUB_AVAILABLE:
            logger.warning("HuggingFace Hub not available - skipping push")
            return
        
        try:
            # Create temporary repository for this checkpoint
            temp_repo_name = f"{self.hf_repo_name}-{checkpoint_name}"
            
            # Upload files to Hub
            self.hf_api.upload_folder(
                folder_path=checkpoint_path,
                repo_id=temp_repo_name,
                repo_type="model",
                private=self.hf_private
            )
            
            logger.info(f"Checkpoint pushed to HuggingFace Hub: {temp_repo_name}")
            
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace Hub: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, model_class, tokenizer_class):
        """
        Load model checkpoint from saved path.
        
        Args:
            checkpoint_path: Path to checkpoint
            model_class: Model class to load
            tokenizer_class: Tokenizer class to load
            
        Returns:
            Tuple of (model, tokenizer, metadata)
        """
        try:
            # Load model and tokenizer
            model = model_class.from_pretrained(checkpoint_path)
            tokenizer = tokenizer_class.from_pretrained(checkpoint_path)
            
            # Load metadata
            metadata_path = os.path.join(checkpoint_path, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            
            logger.info(f"Checkpoint loaded successfully: {checkpoint_path}")
            return model, tokenizer, metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        # List local checkpoints
        if os.path.exists(self.local_backup_path):
            for file in os.listdir(self.local_backup_path):
                if file.endswith('.zip'):
                    checkpoint_info = {
                        "name": file.replace('.zip', ''),
                        "path": os.path.join(self.local_backup_path, file),
                        "type": "local_backup",
                        "size": os.path.getsize(os.path.join(self.local_backup_path, file))
                    }
                    checkpoints.append(checkpoint_info)
        
        # List Google Drive checkpoints
        if COLAB_AVAILABLE and os.path.exists(self.google_drive_path):
            for file in os.listdir(self.google_drive_path):
                if file.endswith('.zip'):
                    checkpoint_info = {
                        "name": file.replace('.zip', ''),
                        "path": os.path.join(self.google_drive_path, file),
                        "type": "google_drive",
                        "size": os.path.getsize(os.path.join(self.google_drive_path, file))
                    }
                    checkpoints.append(checkpoint_info)
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[keep_last:]:
            try:
                os.remove(checkpoint["path"])
                logger.info(f"Removed old checkpoint: {checkpoint['name']}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint['name']}: {e}")
    
    def download_from_hub(self, checkpoint_name: str, local_path: str) -> bool:
        """
        Download checkpoint from HuggingFace Hub.
        
        Args:
            checkpoint_name: Name of checkpoint on Hub
            local_path: Local path to save checkpoint
            
        Returns:
            True if download successful
        """
        if not HF_HUB_AVAILABLE:
            logger.error("HuggingFace Hub not available")
            return False
        
        try:
            # Download from Hub
            self.hf_api.snapshot_download(
                repo_id=f"{self.hf_repo_name}-{checkpoint_name}",
                local_dir=local_path,
                repo_type="model"
            )
            
            logger.info(f"Checkpoint downloaded from Hub: {checkpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download from Hub: {e}")
            return False

class CheckpointCallback:
    """
    Training callback for automatic checkpoint management.
    
    Integrates with HuggingFace Trainer to provide automatic
    checkpoint saving and backup.
    """
    
    def __init__(self, 
                 checkpoint_manager: CheckpointManager,
                 save_interval: int = 500,
                 backup_interval: int = 1000):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            save_interval: Steps between checkpoint saves
            backup_interval: Steps between backups
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_interval
        self.backup_interval = backup_interval
        self.step_count = 0
    
    def on_save(self, args, state, control, **kwargs):
        """Called when checkpoint is saved."""
        self.step_count = state.global_step
        
        if self.step_count % self.backup_interval == 0:
            logger.info(f"Creating backup at step {self.step_count}")
            # Additional backup logic can be added here
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info("Training completed - final checkpoint management")
        # Cleanup old checkpoints
        self.checkpoint_manager.cleanup_old_checkpoints()

# Convenience functions
def create_checkpoint_manager(hf_repo_name: str = "therabot-llama-3.1-8b-therapy-lora",
                            hf_private: bool = True) -> CheckpointManager:
    """Create a checkpoint manager instance."""
    return CheckpointManager(hf_repo_name, hf_private)

def create_checkpoint_callback(checkpoint_manager: CheckpointManager,
                             save_interval: int = 500) -> CheckpointCallback:
    """Create a checkpoint callback for training."""
    return CheckpointCallback(checkpoint_manager, save_interval)
