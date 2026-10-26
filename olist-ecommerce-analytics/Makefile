.PHONY: setup data warehouse report dashboard test all
setup:      ; pip install -r requirements.txt
data:       ; python -m src.download_data
warehouse:  ; python -m src.build_warehouse
report:     ; python -m src.make_report
dashboard:  ; streamlit run dashboard/app.py
test:       ; pytest -q
all: data warehouse report
