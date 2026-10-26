.PHONY: setup setup-neural eval eval-neural demo test
setup:        ; pip install -r requirements.txt pytest ruff
setup-neural: ; pip install -r requirements-neural.txt pytest ruff
eval:         ; cd src && python -m scifact.evaluate
eval-neural:  ; cd src && python -m scifact.evaluate --neural
demo:         ; streamlit run app/demo.py
test:         ; pytest -q
