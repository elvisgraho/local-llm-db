import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# -------------------------------------------------------------------------
# 1. Project Setup & Dependencies
# -------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Reusing specific dependencies from your codebase
from training.processing_utils import get_unique_path
from training.history_manager import ProcessingHistory

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("readme_aggregation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 2. Configuration
# -------------------------------------------------------------------------

IGNORED_DIRS = {
    '.git', 'node_modules', 'target', 'dist', 'build', 
    '__pycache__', 'venv', '.idea', '.vscode', '.next', 'bin', 'obj', '.github', 'theme'
}

IGNORED_FILES = {
    '.DS_Store', 'package-lock.json', 'yarn.lock', 
    'Cargo.lock', 'poetry.lock', 'LICENSE', '.gitignore', 'robots.txt'
}

# Map extensions to markdown language tags
EXT_TO_LANG = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.rs': 'rust',
    '.go': 'go', '.java': 'java', '.c': 'c', '.cpp': 'cpp', '.h': 'cpp',
    '.sh': 'bash', '.md': 'markdown', '.json': 'json', '.yml': 'yaml',
    '.toml': 'toml', '.sql': 'sql', '.html': 'html',
    '.dockerfile': 'dockerfile', '.bat': 'batch', '.ps1': 'powershell'
}

def get_language_tag(filepath: Path) -> str:
    return EXT_TO_LANG.get(filepath.suffix.lower(), '')

def is_text_file(filepath: Path) -> bool:
    """Read start of file to ensure it's not binary."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, Exception):
        return False

# -------------------------------------------------------------------------
# 3. Aggregation Logic
# -------------------------------------------------------------------------

def determine_target_group(file_rel_path: Path, max_depth: int) -> Path:
    """
    Decides which README folder this file belongs to.
    """
    parts = file_rel_path.parent.parts
    if len(parts) <= max_depth:
        return Path(*parts)
    return Path(*parts[:max_depth])

def generate_markdown_content(group_rel_path: Path, files: List[Path], input_root: Path) -> str:
    """
    Generates structured Markdown.
    It groups files by their top-level subdirectory relative to the README location.
    """
    buffer = []
    
    # Header for the file
    readme_name = group_rel_path if str(group_rel_path) != '.' else 'Root Directory'
    buffer.append(f"# Documentation: {readme_name}")
    buffer.append(f"**Base Path:** `{group_rel_path}`\n")

    # 1. Organize files into Sections based on subfolders
    # Key = Subfolder Name (or '.' for root of this group), Value = List of files
    sections: Dict[str, List[Path]] = defaultdict(list)
    
    full_group_path = input_root / group_rel_path

    for file_path in files:
        try:
            # Get path relative to where the README sits
            rel_to_group = file_path.relative_to(full_group_path)
            
            # If it's deep (e.g., Amsi_HBP/src/main.rs), the section is 'Amsi_HBP'
            # If it's shallow (e.g., build.sh), the section is '.'
            parts = rel_to_group.parts
            if len(parts) > 1:
                section_key = parts[0]
            else:
                section_key = "."
                
            sections[section_key].append(file_path)
        except ValueError:
            continue

    # 2. Process Sections (Sorted alphabetically)
    # We explicitly handle '.' (files at the root of this readme) first
    sorted_sections = sorted(sections.keys(), key=lambda x: (x != '.', x.lower()))

    for section in sorted_sections:
        section_files = sections[section]
        
        # Section Header
        if section != ".":
            buffer.append(f"\n# Module: {section}")
            buffer.append(f"---")
        else:
            if len(sections) > 1:
                buffer.append(f"\n# Base Files")

        # Sort files: READMEs first, then alphabetical
        section_files.sort(key=lambda p: (0 if p.name.lower().startswith('readme') else 1, p.name))

        for file_path in section_files:
            if not is_text_file(file_path):
                continue

            try:
                # Calculate display path relative to the section
                # e.g., if we are in section 'Amsi_HBP', file 'Amsi_HBP/src/main.rs' -> 'src/main.rs'
                rel_to_group = file_path.relative_to(full_group_path)
                if section != ".":
                    display_path = rel_to_group.relative_to(section)
                else:
                    display_path = rel_to_group

                content = file_path.read_text(encoding='utf-8', errors='replace').strip()
                if not content: continue

                # -- Render logic --
                if file_path.name.lower().startswith('readme'):
                    # Render READMEs as text
                    buffer.append(f"\n### 📄 {display_path}\n")
                    buffer.append(content)
                    buffer.append("\n")
                else:
                    # Render Code as blocks
                    lang = get_language_tag(file_path)
                    buffer.append(f"\n### 📎 `{display_path}`")
                    buffer.append(f"```{lang}")
                    buffer.append(content)
                    buffer.append("```\n")

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")

    return "\n".join(buffer)

# -------------------------------------------------------------------------
# 4. Main Execution
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate code files into structured READMEs.")
    parser.add_argument("input_dir", type=Path, help="Path to source folder")
    parser.add_argument("output_dir", type=Path, nargs="?", 
                        default=Path(__file__).parent / "processed_readmes", help="Output folder")
    parser.add_argument("--depth", type=int, default=1, 
                        help="Grouping depth. 0=Root Only (All files in one readme), 1=Separate readme for top-level folders, etc.")
    parser.add_argument("--force", action="store_true", help="Ignore history.")
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"CRITICAL: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # History Setup
    history_path = args.output_dir / "processing_history.json"
    if args.force and history_path.exists():
        history_path.unlink()
    history = ProcessingHistory(history_path)

    print(f"--- Configuration ---")
    print(f"Source: {args.input_dir.resolve()}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Depth:  {args.depth}")

    # 1. Scan and Group Files
    file_groups: Dict[Path, List[Path]] = defaultdict(list)
    total_files = 0

    for root, dirs, files in os.walk(args.input_dir):
        # Filter directories to skip
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith('.')]
        
        for file in files:
            if file in IGNORED_FILES or file.startswith('.'):
                continue

            abs_path = Path(root) / file
            rel_path = abs_path.relative_to(args.input_dir)
            
            target_group = determine_target_group(rel_path, args.depth)
            file_groups[target_group].append(abs_path)
            total_files += 1

    print(f"Found {total_files} files across {len(file_groups)} target READMEs.\n")

    # 2. Generate Content
    stats = {"processed": 0, "errors": 0}

    for group_rel_path, files in file_groups.items():
        # Destination: output_dir / group_rel_path / README.md
        target_dir = args.output_dir / group_rel_path
        target_file = target_dir / "README.md"

        if not args.force and history.should_process(target_file):
            # Using target_file as key for history since it represents the aggregate state
            continue

        try:
            content = generate_markdown_content(group_rel_path, files, args.input_dir)
            
            if not content.strip():
                continue

            target_dir.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding='utf-8')
            
            # Record history
            for src_file in files:
                history.record_processing(src_file, output_file=str(target_file.resolve()))
            
            logger.info(f"Generated: {target_file}")
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"Failed to write {target_file}: {e}")
            stats["errors"] += 1

    history.save()

    print(f"\n--- Summary ---")
    print(f"READMEs Created: {stats['processed']}")
    print(f"Errors:          {stats['errors']}")

if __name__ == "__main__":
    main()