"""
Base class for batch document processing operations.
Consolidates common patterns across batch_parser, batch_writeup, and batch_tagger.
"""
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from langchain_core.documents import Document

from training.load_documents import load_documents
from training.history_manager import ProcessingHistory
from training.processing_utils import calculate_context_ceiling

logger = logging.getLogger(__name__)


class BatchProcessorBase(ABC):
    """Abstract base class for batch document processors."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        force: bool = False,
        supported_extensions: Optional[set] = None
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.force = force
        self.supported_extensions = supported_extensions or {'.pdf', '.txt', '.md', '.markdown'}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        history_path = output_dir / self.get_history_filename()
        if force and history_path.exists():
            history_path.unlink()
        self.history = ProcessingHistory(history_path)

        self.stats = {"processed": 0, "errors": 0, "skipped": 0}

    @abstractmethod
    def get_history_filename(self) -> str:
        """Return the history file name for this processor."""
        pass

    @abstractmethod
    def process_document(self, doc: Document, index: int, total: int) -> Optional[Dict[str, Any]]:
        """
        Process a single document.
        Returns dict with 'output_path' and optional metadata, or None to skip.
        """
        pass

    def get_pending_paths(self) -> List[Path]:
        """Scan directory for files that need processing."""
        logger.info("Scanning directory for pending files...")

        all_paths = (p.resolve() for p in self.input_dir.rglob("*") if p.is_file())

        pending_paths = [
            p for p in all_paths
            if p.suffix.lower() in self.supported_extensions and self.history.should_process(p)
        ]

        return pending_paths

    def load_pending_documents(self, pending_paths: List[Path]) -> List[Document]:
        """Load documents from pending file paths."""
        if not pending_paths:
            logger.info("All documents already processed. Use --force to override.")
            return []

        logger.info(f"Loading content for {len(pending_paths)} pending files...")
        try:
            return load_documents(file_paths=pending_paths)
        except Exception as e:
            logger.error(f"Loader failed: {e}")
            return []

    def optimize_documents(self, documents: List[Document]) -> List[Document]:
        """Optimize document ordering for resource allocation."""
        return calculate_context_ceiling(documents)

    def run(self):
        """Execute the batch processing pipeline."""
        if not self.input_dir.exists():
            logger.error(f"CRITICAL: Input directory '{self.input_dir}' does not exist.")
            sys.exit(1)

        pending_paths = self.get_pending_paths()

        if not pending_paths:
            logger.info("No pending documents to process.")
            return

        documents = self.load_pending_documents(pending_paths)

        if not documents:
            logger.info("No documents were successfully loaded.")
            return

        documents = self.optimize_documents(documents)

        logger.info(f"--- Configuration ---")
        logger.info(f"Pending Documents: {len(documents)}")
        logger.info(f"Target: {self.output_dir}")
        logger.info(f"Starting processing...\n")

        try:
            for i, doc in enumerate(documents, 1):
                if self._should_stop():
                    break

                source_path = Path(doc.metadata.get('source', '')).resolve()

                try:
                    result = self.process_document(doc, i, len(documents))

                    if result is None:
                        self.stats["skipped"] += 1
                        continue

                    output_path = result.get('output_path')
                    extra_metadata = {k: v for k, v in result.items() if k != 'output_path'}

                    self.history.record_processing(
                        source_path,
                        output_file=str(output_path) if output_path else None,
                        **extra_metadata
                    )
                    self.history.save()
                    self.stats["processed"] += 1

                except Exception as e:
                    logger.error(f"Processing failure for {source_path.name}: {e}")
                    self.stats["errors"] += 1

        except KeyboardInterrupt:
            logger.info("\n\n[!] Interrupt detected. State synchronized.")
        finally:
            self.history.save()

        self._print_summary()

    def _should_stop(self) -> bool:
        """Check if processing should stop. Override for custom behavior."""
        return False

    def _print_summary(self):
        """Print processing summary."""
        logger.info(f"\n--- Summary ---")
        logger.info(f"Successfully Processed: {self.stats['processed']}")
        logger.info(f"Errors:                {self.stats['errors']}")
        logger.info(f"Skipped:               {self.stats['skipped']}")
