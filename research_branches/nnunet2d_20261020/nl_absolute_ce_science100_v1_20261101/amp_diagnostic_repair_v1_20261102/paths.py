import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from common import GROUP
OUT = GROUP / HERE.name
