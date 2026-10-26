.PHONY: setup train serve test docker
setup:  ; pip install -r requirements-dev.txt
train:  ; cd src && python -m churn.train
serve:  ; cd src && uvicorn churn.api:app --reload
test:   ; pytest -q
docker: ; docker build -t churn-api . && docker run -p 8000:8000 churn-api
