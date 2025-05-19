.PHONY: setup run dev install test

# Set the Python version from cookiecutter or default to 3.12
PYTHON_VERSION := 3.12
TRANSPORT_MODE ?= "stdio"

# Setup with uv
setup:
	# Check if uv is installed, install if not
	@which uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
	# Create a virtual environment
	uv venv
	# Install dependencies with development extras
	uv pip install -e ".[dev]"
	@echo "✅ Environment setup complete. Activate it with 'source .venv/bin/activate' (Unix/macOS) or '.venv\\Scripts\activate' (Windows)"

# Run the server directly
run:
	uv run src/alertmanager_mcp_server/server.py --transport ${TRANSPORT_MODE}

# Run in development mode with MCP inspector
dev:
	mcp dev src/alertmanager_mcp_server/server.py

# Install in Claude desktop
install:
	mcp install src/alertmanager_mcp_server/server.py

# Docker build
docker-build:
	docker build -t kiennt26/alertmanager_mcp_server:latest .

# Run with Docker
docker-run:
	docker run -p 8000:8000 kiennt26/alertmanager_mcp_server:latest

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
