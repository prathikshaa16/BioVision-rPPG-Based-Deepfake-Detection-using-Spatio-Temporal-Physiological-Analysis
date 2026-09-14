"""Prepare reproducible Kaggle artifacts for download as notebook outputs.

Run this inside the Kaggle notebook after training and official evaluation.
It copies, without modifying, the best checkpoint, histories, manifests,
metrics, ROC points, confusion matrix, and raw predictions into one output
folder that Kaggle can publish.
"""

from pathlib import Path
import shutil


WORKING_ROOT = Path('/kaggle/working')
RELEASE_ROOT = WORKING_ROOT / 'BioVision_Release'

SOURCES = {
    'checkpoint': WORKING_ROOT / 'BioVision_Final_Training' / 'biovision_best.pt',
    'training_log': WORKING_ROOT / 'BioVision_Final_Training' / 'training_log.csv',
    'training_summary': WORKING_ROOT / 'BioVision_Final_Training' / 'training_summary.json',
    'test_metrics': WORKING_ROOT / 'BioVision_Final_Evaluation' / 'official_test_metrics.json',
    'test_predictions': WORKING_ROOT / 'BioVision_Final_Evaluation' / 'official_test_predictions.csv',
    'test_roc': WORKING_ROOT / 'BioVision_Final_Evaluation' / 'official_test_roc.csv',
    'test_confusion_matrix': WORKING_ROOT / 'BioVision_Final_Evaluation' / 'official_test_confusion_matrix.csv',
    'test_manifest': WORKING_ROOT / 'BioVision_Official_Test_Final' / 'official_test_manifest_518.csv',
    'validation_report': WORKING_ROOT / 'BioVision_Final_Cache' / 'validation_report.json',
    'split_manifest': WORKING_ROOT / 'BioVision_Final_Cache' / 'identity_aware_split' / 'identity_aware_split_manifest.csv',
}

RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
missing = []
for name, source in SOURCES.items():
    if not source.exists():
        missing.append(f'{name}: {source}')
        continue
    shutil.copy2(source, RELEASE_ROOT / source.name)

if missing:
    raise FileNotFoundError('Missing release artifacts:\n' + '\n'.join(missing))

print(f'Release directory: {RELEASE_ROOT}')
for path in sorted(RELEASE_ROOT.iterdir()):
    print(f'{path.name}: {path.stat().st_size} bytes')
