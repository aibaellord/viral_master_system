#!/usr/bin/env python3
import os
import logging
import shutil
import re
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_backup(file_path):
    """Create a backup of the original file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise

def rollback(original_path, backup_path):
    """Restore the original file from backup if needed"""
    try:
        logger.warning(f"Rolling back changes from backup: {backup_path}")
        shutil.copy2(backup_path, original_path)
        logger.info(f"Rollback successful")
        return True
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        print(f"{Fore.RED}CRITICAL: Both original file and backup may be compromised! {e}{Style.RESET_ALL}")
        return False

def fix_indentation(file_path):
    """Fix the indentation issue in the viral_orchestrator.py file"""
    backup_path = None
    try:
        # Verify file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {file_path}")
        
        # Create backup
        backup_path = create_backup(file_path)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
        
        logger.info(f"File loaded with {len(content)} lines")
        
        # Find the scale_resources method (around line 236)
        in_scale_resources = False
        scale_resources_start = None
        scale_resources_indentation = None
        fixed_content = []
        scale_resources_lines_fixed = 0
        
        for i, line in enumerate(content):
            line_num = i + 1
            
            # Detect start of scale_resources method
            if "scale_resources" in line and (line.strip().startswith("async def scale_resources") or line.strip().startswith("def scale_resources")):
                logger.info(f"Found scale_resources method at line {line_num}: {line.strip()}")
                in_scale_resources = True
                scale_resources_start = i
                # Calculate current indentation (should be 8 spaces)
                scale_resources_indentation = len(line) - len(line.lstrip())
                # Fix indentation for method definition (should be 4 spaces)
                fixed_line = "    " + line[scale_resources_indentation:]
                fixed_content.append(fixed_line)
                scale_resources_lines_fixed += 1
                continue
            
            # Fix indentation for all lines within scale_resources method
            if in_scale_resources:
                # Check if we've exited the method (empty line or next method)
                if line.strip() and not line.lstrip().startswith("#") and scale_resources_indentation is not None:
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= 4 and ("def " in line or "async def " in line) and (line.lstrip().startswith("def ") or line.lstrip().startswith("async def ")):
                        # We've reached the next method
                        in_scale_resources = False
                        fixed_content.append(line)
                        continue
                
                if scale_resources_indentation is not None and line.strip():
                    # Adjust indentation for non-empty lines in the method
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent >= scale_resources_indentation:
                        # Reduce indentation by 4 spaces (from 8 to 4 for the first level)
                        fixed_line = "    " + line[scale_resources_indentation:]
                        fixed_content.append(fixed_line)
                        scale_resources_lines_fixed += 1
                        continue
                
                # If we reach here, either it's an empty line or we've exited the method
                if not line.strip():
                    fixed_content.append(line)
                    continue
                
                # If we reach a new method or indentation changes, mark as no longer in scale_resources
                in_scale_resources = False
                fixed_content.append(line)
            else:
                # Regular line, no changes needed
                fixed_content.append(line)
        
        logger.info(f"Adjusted indentation for {scale_resources_lines_fixed} lines in scale_resources method")
        
        # Verify changes were made
        if scale_resources_start is None:
            raise ValueError("Could not find 'scale_resources' method in the file")
        
        if scale_resources_lines_fixed == 0:
            raise ValueError("No lines were fixed in the 'scale_resources' method")
        
        # Write the corrected content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(fixed_content)
        
        logger.info(f"Successfully fixed indentation in {file_path}")
        print(f"{Fore.GREEN}Successfully fixed indentation issues in {file_path}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Backup created at: {backup_path}{Style.RESET_ALL}")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing indentation: {e}")
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        
        # Attempt rollback if a backup was created and there was an error
        if backup_path:
            rollback_success = rollback(file_path, backup_path)
            if rollback_success:
                print(f"{Fore.YELLOW}Changes rolled back to original state{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to roll back changes{Style.RESET_ALL}")
        
        return False

def main():
    """Main function to run the script"""
    file_path = "core/engine/viral_orchestrator.py"
    
    try:
        print(f"{Fore.CYAN}Starting indentation fix for {file_path}...{Style.RESET_ALL}")
        fix_indentation(file_path)
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        print(f"{Fore.YELLOW}Script interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        print(f"{Fore.RED}Critical error: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

