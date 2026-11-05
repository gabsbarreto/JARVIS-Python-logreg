"""Run a single dataset using adaptive logistic-regression weights."""
from pathlib import Path
import sys

# Choose the dataset when running directly in Spyder. See README for choices.
DATASET = 'portionsize'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import main

if __name__ == '__main__':
    main('adaptive', default_dataset=DATASET)
