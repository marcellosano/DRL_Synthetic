#!/bin/bash
# Launch the DRL Coastal EWS Dashboard

echo "🌊 Starting DRL Coastal Emergency Warning System Dashboard..."
echo "=================================================="
echo ""

cd /home/msano/Projects/DRL_Synthetic

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip3 install -r requirements.txt --break-system-packages
fi

echo "🚀 Launching dashboard on http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py