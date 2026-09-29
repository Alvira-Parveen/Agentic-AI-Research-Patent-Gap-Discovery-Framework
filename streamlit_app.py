import runpy
from pathlib import Path

# Entry point for Streamlit Cloud root deployment
app_path = Path(__file__).parent / "app" / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
