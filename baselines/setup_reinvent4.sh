#!/bin/bash
# Setup script for REINVENT 4 in the nih_research conda environment
#
# This script:
# 1. Installs REINVENT 4 dependencies (without reinstalling PyTorch)
# 2. Downloads prior models from Zenodo
# 3. Verifies the installation
#
# Usage: bash setup_reinvent4.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REINVENT4_DIR="$(cd "$SCRIPT_DIR/../../REINVENT4-main" && pwd)"
PRIORS_DIR="$REINVENT4_DIR/priors"

echo "============================================"
echo "  REINVENT 4 Setup for nih_research env"
echo "============================================"
echo ""
echo "REINVENT4 source: $REINVENT4_DIR"
echo "Priors directory: $PRIORS_DIR"
echo ""

# Activate conda environment
echo "[1/4] Activating conda environment..."
source activate nih_research 2>/dev/null || conda activate nih_research

# Install REINVENT4 in editable mode, skipping PyTorch (already installed)
# We use pip install -e with --no-deps first to register the package,
# then install missing dependencies individually
echo "[2/4] Installing REINVENT 4 (editable, no optional deps)..."
cd "$REINVENT4_DIR"

# Install without forcing PyTorch version - use --no-build-isolation to
# avoid rebuilding with a different torch
pip install -e "." \
    --no-deps \
    2>&1 | tail -5

# Install missing dependencies that are not already present
echo ""
echo "[2b/4] Installing missing dependencies..."
pip install --quiet \
    "funcy>=2,<3" \
    "molvs>=0.1.1,<0.2" \
    "pydantic>=2,<3" \
    "python-dotenv>=1.0,<2" \
    "tomli>=2.0,<3" \
    "tenacity>=8.2,<9" \
    "tensorboard>=2,<3" \
    "xxhash>=3,<4" \
    "apted>=1.0.3,<2" \
    "typing_extensions>=4.0,<5" \
    "polars<2" \
    "pumas>=1.3.0" \
    2>&1 | tail -5

# Optional: install chemprop and descriptastorus (may have conflicts)
echo ""
echo "[2c/4] Installing chemprop (optional, for QSAR scoring)..."
pip install --quiet "chemprop>=1.5.2,<1.6" "descriptastorus>=2.6.1,<3.0" 2>&1 | tail -3 || \
    echo "  WARNING: chemprop install failed (optional, needed only for QSAR scoring)"

# Optional: install mmpdb
echo ""
pip install --quiet "mmpdb>=2.1,<3" 2>&1 | tail -3 || \
    echo "  WARNING: mmpdb install failed (optional)"

# Download prior models if not present
echo ""
echo "[3/4] Checking prior models..."
mkdir -p "$PRIORS_DIR"

if [ ! -f "$PRIORS_DIR/reinvent.prior" ]; then
    echo "  Prior models not found. Downloading from Zenodo..."
    echo "  URL: https://doi.org/10.5281/zenodo.15641296"
    echo ""
    echo "  Please download the prior models manually:"
    echo "    1. Visit: https://zenodo.org/records/15641296"
    echo "    2. Download reinvent.prior (and optionally other priors)"
    echo "    3. Place them in: $PRIORS_DIR/"
    echo ""
    echo "  Alternatively, run:"
    echo "    cd $PRIORS_DIR"
    echo "    wget https://zenodo.org/records/15641296/files/reinvent.prior"
    echo ""
    echo "  NOTE: Without prior models, REINVENT 4 RL/sampling cannot run."
    echo "        The baseline script will still be importable for testing."
else
    echo "  Prior models found at $PRIORS_DIR/"
    ls -la "$PRIORS_DIR"/*.prior 2>/dev/null
fi

# Verify installation
echo ""
echo "[4/4] Verifying installation..."
python -c "
import sys
sys.path.insert(0, '$REINVENT4_DIR')

try:
    import reinvent
    print(f'  REINVENT version: {reinvent.__version__}')
except ImportError as e:
    print(f'  ERROR: Cannot import reinvent: {e}')
    sys.exit(1)

try:
    import torch
    print(f'  PyTorch version:  {torch.__version__}')
    print(f'  CUDA available:   {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU device:       {torch.cuda.get_device_name(0)}')
except ImportError:
    print('  WARNING: PyTorch not available')

try:
    from rdkit import Chem
    import rdkit
    print(f'  RDKit version:    {rdkit.__version__}')
except ImportError:
    print('  WARNING: RDKit not available')

print()
print('  Installation verification complete.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To run the baseline:"
echo "  python $SCRIPT_DIR/reinvent4_baseline.py \\"
echo "    --dataset scd1 \\"
echo "    --data_path /path/to/scd1_binding.csv \\"
echo "    --output_dir /path/to/results"
echo ""
