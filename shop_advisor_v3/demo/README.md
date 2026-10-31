# Demo pack

- `files/`: 23 SYNTHETIC upload files, one per situation (12 business situations, 7 messy-file cases, 4 files the system must refuse).
- `DEMO_GUIDE.md`: for every file, what to type on the website, what is in the data, what the system measured, what it concluded, which rules fired, and a sentence to say. Generated from real runs of the system, not written by hand.
- `inputs.json`: the same business details as JSON.
- `make_demo.py`: rebuilds the files and the guide: `python demo/make_demo.py`.

Run the website: `uvicorn startupiq.api.app:app`, open http://127.0.0.1:8000, upload a file from `files/`, type the details from the guide, press Analyse.

Suggested 5-minute order: 02 (stock-out before the festival), 03 (growing but out of cash), 06 (price war on a thin margin), 05 (losing ground), 13 (messy Excel), 20 (refused file).
