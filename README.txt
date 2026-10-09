ISRO AI MISSION CONTROL — FINAL EXHIBITION EDITION

This version preserves every field from the original project brief:
Rocket, Launch Site, Payload, Mission Type, Target Orbit, Launch Year,
Mission Parameters, predicted launch outcome, estimated mission cost,
rocket, confidence, key factors, historical analysis, feature importance,
interactive dashboard and technology stack.

It also adds:
- Dramatic spaceport/launch-control background
- All major ISRO launcher families and historical variants
- FLP and SLP site options
- Future Kulasekarapattinam option kept separate from historical data
- Official ISRO launch records
- Robust error handling and a Windows launcher
- Honest labels for non-official cost estimates

RUN:
1. Install Python 3.10 or newer.
2. Extract this ZIP.
3. Double-click run_project.bat.

OR:
py -m pip install -r requirements.txt
py -m streamlit run app.py

If Windows says "py is not recognized", use:
python -m pip install -r requirements.txt
python -m streamlit run app.py

DATA:
Historical launch records are from ISRO's official Launch Missions listing.
No synthetic historical missions are added.
Cost estimation is clearly marked illustrative because ISRO does not publish
a complete per-launch cost series in the Launch Missions table.
