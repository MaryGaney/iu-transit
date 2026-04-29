#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-shot backend setup for IU Transit Tracker
# Run from the backend/ directory: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PYTHON=${PYTHON:-python3}
VENV=".venv"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " IU Transit Tracker — Backend Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python version
$PYTHON --version
PYVER=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PYVER" -lt 11 ]; then
  echo "❌ Python 3.11+ required. Found Python 3.$PYVER"
  exit 1
fi
echo "✅ Python OK"

# Create virtualenv
if [ ! -d "$VENV" ]; then
  echo ""
  echo "Creating virtual environment..."
  $PYTHON -m venv $VENV
fi
source $VENV/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies (this takes ~2 minutes first time)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✅ Dependencies installed"

# Create .env if not present
if [ ! -f ".env" ]; then
  echo ""
  echo "Creating .env from template..."
  cp .env.example .env
  echo "⚠️  Edit .env and set your MAPBOX_TOKEN before starting the server."
fi

# Create data directory
mkdir -p data
echo "✅ data/ directory ready"

# Probe GTFS-RT endpoints
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Probing Bloomington Transit GTFS-RT endpoints..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON scripts/probe_gtfs_rt.py || echo "⚠️  RT probe failed — you can run it manually later: python scripts/probe_gtfs_rt.py"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Setup complete. Next steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Edit .env and set MAPBOX_TOKEN and correct GTFS_RT_VEHICLE_URL"
echo ""
echo "  2. Start the backend:"
echo "     source .venv/bin/activate && uvicorn app.main:app --reload"
echo ""
echo "  3. Upload the IU class schedule CSV:"
echo "     curl -X POST http://localhost:8000/api/admin/load-schedule \\"
echo "          -F 'file=@/path/to/your/schedule.csv'"
echo ""
echo "  4. Check system status:"
echo "     curl http://localhost:8000/api/admin/status | python3 -m json.tool"
echo ""
echo "  5. View API docs: http://localhost:8000/api/docs"
echo ""
