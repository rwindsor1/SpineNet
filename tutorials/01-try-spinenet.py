import sys
import os
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
print(os.environ.get("PYTHONPATH"))
import spinenet

print(spinenet)

# spnt = spinenet.SpineNet()
