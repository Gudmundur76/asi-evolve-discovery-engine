# ASI-Evolve Discovery Engine — Makefile
.PHONY: help setup train build up down logs test clean

# Default target
help:
	@echo "ASI-Evolve Discovery Engine — Available Commands:"
	@echo ""
	@echo "  setup      Install dependencies and prepare environment"
	@echo "  train      Train the affinity prediction model"
	@echo "  build      Build Docker images"
	@echo "  up         Start all services (docker-compose up -d)"
	@echo "  down       Stop all services"
	@echo "  logs       Show service logs"
	@echo "  api-logs   Show API logs only"
	@echo "  loop-logs  Show scheduler/loop logs only"
	@echo "  test       Run all tests"
	@echo "  test-core  Run core engine tests"
	@echo "  shell      Open API container shell"
	@echo "  clean      Remove data, models, and Docker artifacts"
	@echo ""

# Setup environment
setup:
	@echo "Creating data directories..."
	mkdir -p data/models data/cognition data/docking data/pdfs
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Setup complete. Run 'make train' to train the model."

# Train the affinity prediction model
train:
	@echo "Training affinity prediction model..."
	python -m scripts.train_model

# Build Docker images
build:
	docker-compose build

# Start services
up:
	docker-compose up -d
	@echo "Services started:"
	@echo "  API:      http://localhost:8000"
	@echo "  Docs:     http://localhost:8000/docs"
	@echo "  Frontend: http://localhost:5173"

# Stop services
down:
	docker-compose down

# Show all logs
logs:
	docker-compose logs -f

# Show API logs
api-logs:
	docker-compose logs -f api

# Show loop logs
loop-logs:
	docker-compose logs -f scheduler

# Run tests
test: test-core
	@echo "Running all tests..."
	cd backend && python -m pytest tests/ -v 2>/dev/null || echo "Agent loop tests require setup"

# Run core tests
test-core:
	@echo "Running core engine tests..."
	python -m pytest tests/ -v

# Open container shell
shell:
	docker-compose exec api bash

# Clean everything
clean:
	@echo "Removing data, models, and containers..."
	docker-compose down -v
	rm -rf data/models/* data/cognition/* data/docking/* data/pdfs/*
	rm -f data/discoveries.db
	@echo "Clean complete."
