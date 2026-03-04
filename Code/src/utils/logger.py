import sys
import os

class DualLogger(object):
    def __init__(self, log_file_path, original_stream):
        self.terminal = original_stream
        self.log_file_path = log_file_path
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()

def setup_global_logger(base_dir):
    """Intercepts all Python print() and error statements to mirror them into Log.txt"""
    # BASE_DIR is .../Code, so we go up one level to the project root
    log_file = os.path.join(base_dir, '..', 'Log.txt')
    
    # Only override if we haven't already
    if not isinstance(sys.stdout, DualLogger):
        sys.stdout = DualLogger(log_file, sys.stdout)
        sys.stderr = DualLogger(log_file, sys.stderr)
