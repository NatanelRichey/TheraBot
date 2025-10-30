#!/usr/bin/env python3
"""
Upload Checkpoints to Google Drive (Every 50 Steps)

This script monitors the training checkpoint directory and automatically uploads
checkpoints to Google Drive when they are at multiples of 50 steps.

Usage:
    # Run as a standalone script (monitors in background)
    python upload_checkpoints_to_drive.py --output_dir ./therapy-model-checkpoints
    
    # Or import and use programmatically
    from upload_checkpoints_to_drive import upload_checkpoint_at_steps
    
    upload_checkpoint_at_steps(checkpoint_dir="./therapy-model-checkpoints")
"""

import os
import re
import json
import shutil
import zipfile
import logging
import argparse
from pathlib import Path
from typing import List, Set, Optional
from datetime import datetime
import time

# Google Drive integration
try:
    from google.colab import drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CHECKPOINT_DIR = "./therapy-model-checkpoints"
DEFAULT_DRIVE_PATH = "/content/drive/MyDrive/TheraBot/checkpoints"
CHECKPOINT_UPLOAD_STATE_FILE = ".checkpoint_upload_state.json"


class CheckpointUploader:
    """
    Monitor and upload checkpoints to Google Drive (every 50 steps).
    """
    
    def __init__(self,
                 checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                 drive_path: str = DEFAULT_DRIVE_PATH,
                 upload_interval: int = 50):
        """
        Initialize checkpoint uploader.
        
        Args:
            checkpoint_dir: Directory where training saves checkpoints
            drive_path: Path to Google Drive checkpoint storage
            upload_interval: Upload checkpoints at multiples of this step number (default: 50)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.drive_path = Path(drive_path)
        self.upload_interval = upload_interval
        self.state_file = self.checkpoint_dir / CHECKPOINT_UPLOAD_STATE_FILE
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Google Drive path (assuming already mounted)
        if COLAB_AVAILABLE:
            self.drive_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Google Drive path ready: {self.drive_path}")
        else:
            logger.warning("Google Colab not detected - Drive path may not be accessible")
        
        # Load upload state
        self.uploaded_checkpoints: Set[int] = self._load_state()
    
    def _load_state(self) -> Set[int]:
        """Load previously uploaded checkpoint steps."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('uploaded_steps', []))
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return set()
    
    def _save_state(self):
        """Save uploaded checkpoint steps to state file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'uploaded_steps': sorted(list(self.uploaded_checkpoints)),
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state file: {e}")
    
    def _extract_step_number(self, checkpoint_name: str) -> Optional[int]:
        """
        Extract step number from checkpoint directory name.
        
        Expected format: checkpoint-{step_number}
        Example: checkpoint-50 -> 50
        """
        match = re.search(r'checkpoint-(\d+)', checkpoint_name)
        if match:
            return int(match.group(1))
        return None
    
    def _find_checkpoint_dirs(self) -> List[tuple]:
        """
        Find all checkpoint directories and return (step_number, path) tuples.
        
        Returns:
            List of (step_number, checkpoint_path) tuples sorted by step number
        """
        checkpoints = []
        
        if not self.checkpoint_dir.exists():
            return checkpoints
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                step = self._extract_step_number(item.name)
                if step is not None:
                    checkpoints.append((step, item))
        
        return sorted(checkpoints, key=lambda x: x[0])
    
    def _should_upload(self, step: int) -> bool:
        """Check if checkpoint at this step should be uploaded."""
        # Upload if step is multiple of upload_interval
        if step % self.upload_interval != 0:
            return False
        
        # Upload if not already uploaded
        if step in self.uploaded_checkpoints:
            logger.debug(f"Checkpoint at step {step} already uploaded, skipping")
            return False
        
        return True
    
    def _zip_checkpoint(self, checkpoint_path: Path) -> Optional[Path]:
        """Create a zip file of the checkpoint directory."""
        zip_name = f"{checkpoint_path.name}.zip"
        zip_path = self.checkpoint_dir / zip_name
        
        try:
            logger.info(f"Creating zip file: {zip_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(checkpoint_path):
                    for file in files:
                        file_path = Path(root) / file
                        # Get relative path from checkpoint directory
                        arc_path = file_path.relative_to(checkpoint_path)
                        zipf.write(file_path, arc_path)
            
            logger.info(f"Created zip file: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
            return zip_path
            
        except Exception as e:
            logger.error(f"Failed to create zip file for {checkpoint_path}: {e}")
            return None
    
    def _upload_to_drive(self, zip_path: Path, step: int) -> bool:
        """Upload zip file to Google Drive."""
        if not COLAB_AVAILABLE:
            logger.warning("Google Colab not available - skipping Drive upload")
            return False
        
        if not self.drive_path.exists():
            logger.error(f"Google Drive path does not exist: {self.drive_path}")
            logger.info("Attempting to mount Google Drive...")
            try:
                drive.mount('/content/drive')
                self.drive_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to mount Google Drive: {e}")
                return False
        
        try:
            drive_zip_path = self.drive_path / zip_path.name
            logger.info(f"Uploading {zip_path.name} to Google Drive...")
            shutil.copy2(zip_path, drive_zip_path)
            
            # Verify upload
            if drive_zip_path.exists():
                logger.info(f"✅ Successfully uploaded checkpoint-{step} to {drive_zip_path}")
                logger.info(f"   Size: {drive_zip_path.stat().st_size / (1024*1024):.2f} MB")
                
                # Mark as uploaded
                self.uploaded_checkpoints.add(step)
                self._save_state()
                
                # Optionally remove local zip to save space
                # zip_path.unlink()
                
                return True
            else:
                logger.error(f"Upload verification failed: {drive_zip_path} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Failed to upload to Google Drive: {e}")
            return False
    
    def upload_checkpoint(self, step: int, checkpoint_path: Path) -> bool:
        """
        Upload a single checkpoint to Google Drive.
        
        Args:
            step: Step number of the checkpoint
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            True if upload successful
        """
        if not self._should_upload(step):
            return False
        
        logger.info(f"Processing checkpoint at step {step}...")
        
        # Create zip file
        zip_path = self._zip_checkpoint(checkpoint_path)
        if zip_path is None:
            return False
        
        # Upload to Drive
        success = self._upload_to_drive(zip_path, step)
        
        return success
    
    def scan_and_upload(self, dry_run: bool = False) -> int:
        """
        Scan checkpoint directory and upload eligible checkpoints.
        
        Args:
            dry_run: If True, only log what would be uploaded without actually uploading
            
        Returns:
            Number of checkpoints uploaded
        """
        checkpoints = self._find_checkpoint_dirs()
        
        if not checkpoints:
            logger.info(f"No checkpoints found in {self.checkpoint_dir}")
            return 0
        
        logger.info(f"Found {len(checkpoints)} checkpoint directories")
        logger.info(f"Uploading checkpoints at multiples of {self.upload_interval} steps")
        
        uploaded_count = 0
        
        for step, checkpoint_path in checkpoints:
            if self._should_upload(step):
                if dry_run:
                    logger.info(f"[DRY RUN] Would upload checkpoint-{step}")
                else:
                    if self.upload_checkpoint(step, checkpoint_path):
                        uploaded_count += 1
            else:
                if step % self.upload_interval == 0 and step not in self.uploaded_checkpoints:
                    logger.debug(f"Checkpoint at step {step} is eligible but skipped (already uploaded)")
        
        if uploaded_count > 0:
            logger.info(f"✅ Uploaded {uploaded_count} checkpoint(s) to Google Drive")
        else:
            logger.info("No new checkpoints to upload")
        
        return uploaded_count
    
    def monitor(self, check_interval: int = 60, max_iterations: Optional[int] = None):
        """
        Continuously monitor checkpoint directory and upload new checkpoints.
        
        Args:
            check_interval: Seconds between directory scans (default: 60)
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting checkpoint monitor (every {check_interval}s)")
        logger.info(f"Will upload checkpoints at multiples of {self.upload_interval} steps")
        logger.info(f"Monitoring: {self.checkpoint_dir}")
        logger.info(f"Destination: {self.drive_path}")
        logger.info("Press Ctrl+C to stop")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"\n--- Scan #{iteration} ---")
                
                self.scan_and_upload()
                
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached maximum iterations ({max_iterations})")
                    break
                
                if check_interval > 0:
                    time.sleep(check_interval)
                    
        except KeyboardInterrupt:
            logger.info("\n⚠️  Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            raise


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Upload training checkpoints to Google Drive (every 50 steps)"
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f'Directory containing checkpoints (default: {DEFAULT_CHECKPOINT_DIR})'
    )
    parser.add_argument(
        '--drive-path',
        type=str,
        default=DEFAULT_DRIVE_PATH,
        help=f'Google Drive destination path (default: {DEFAULT_DRIVE_PATH})'
    )
    parser.add_argument(
        '--upload-interval',
        type=int,
        default=50,
        help='Upload checkpoints at multiples of this step number (default: 50)'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Continuously monitor for new checkpoints'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=60,
        help='Seconds between scans when monitoring (default: 60)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    uploader = CheckpointUploader(
        checkpoint_dir=args.checkpoint_dir,
        drive_path=args.drive_path,
        upload_interval=args.upload_interval
    )
    
    if args.monitor:
        uploader.monitor(check_interval=args.check_interval)
    else:
        uploader.scan_and_upload(dry_run=args.dry_run)


class DriveUploadCallback:
    """
    HuggingFace Trainer callback to automatically upload checkpoints to Google Drive.
    
    This callback integrates directly into the training loop and uploads checkpoints
    at specified step intervals (e.g., every 50 steps) without needing a separate thread.
    """
    
    def __init__(self,
                 drive_path: str = DEFAULT_DRIVE_PATH,
                 upload_interval: int = 50):
        """
        Initialize the callback.
        
        Args:
            drive_path: Path to Google Drive checkpoint storage
            upload_interval: Upload checkpoints at multiples of this step number (default: 50)
        """
        self.drive_path = drive_path
        self.upload_interval = upload_interval
        self.uploaded_steps: Set[int] = set()
        self.output_dir = None
        self.uploader = None  # Will be initialized in on_save when we have output_dir
    
    def __getattr__(self, name):
        """Return a no-op function for any callback methods we don't implement."""
        if name.startswith('on_'):
            def no_op(*args, **kwargs):
                pass
            return no_op
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def on_save(self, args, state, control, **kwargs):
        """
        Called whenever a checkpoint is saved.
        Uploads the checkpoint if it's at a multiple of upload_interval.
        """
        step = state.global_step
        
        # Initialize uploader on first save
        if self.uploader is None:
            self.output_dir = Path(args.output_dir)
            self.uploader = CheckpointUploader(
                checkpoint_dir=str(self.output_dir),
                drive_path=self.drive_path,
                upload_interval=self.upload_interval
            )
            self.uploaded_steps = self.uploader._load_state()
        
        # Check if we should upload this checkpoint
        if step % self.upload_interval == 0 and step not in self.uploaded_steps:
            checkpoint_path = self.output_dir / f"checkpoint-{step}"
            
            if checkpoint_path.exists():
                logger.info(f"\n📤 Uploading checkpoint at step {step} to Google Drive...")
                success = self.uploader.upload_checkpoint(step, checkpoint_path)
                
                if success:
                    self.uploaded_steps.add(step)
                    logger.info(f"✅ Checkpoint-{step} uploaded successfully!")
                else:
                    logger.warning(f"⚠️  Failed to upload checkpoint-{step}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training - upload final checkpoint if needed."""
        if self.uploader is not None and self.output_dir is not None:
            if state.global_step % self.upload_interval == 0:
                final_checkpoint = self.output_dir / f"checkpoint-{state.global_step}"
                if final_checkpoint.exists() and state.global_step not in self.uploaded_steps:
                    logger.info(f"\n📤 Uploading final checkpoint at step {state.global_step}...")
                    self.uploader.upload_checkpoint(state.global_step, final_checkpoint)


if __name__ == "__main__":
    main()

