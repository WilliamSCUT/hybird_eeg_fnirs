from pathlib import Path

def load(config):
    path = Path(config['defaults']['data_loader']['path'])
    # load files
    stroop_files = [p for p in path.iterdir() if p.is_dir() and 'stroop' in str(p)]
    mental_calculation_files = [p for p in path.iterdir() if p.is_dir() and 'MentalCalculation' in str(p)]
    digit_span_files = [p for p in path.iterdir() if p.is_dir() and 'digitSpan' in str(p)]



    return None