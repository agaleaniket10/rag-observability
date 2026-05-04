.PHONY: install run test stack-up stack-down clean

install:
	pip install -r requirements.txt

stack-up:
	docker compose up -d

stack-down:
	docker compose down

run:
	python -m src.main

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
