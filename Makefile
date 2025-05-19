.PHONY: setup run dev install deploy test clean format type-check

# Set the Python version from cookiecutter or default to 3.12
PYTHON_VERSION := 3.12

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
	uv run src/alertmanager-mcp-server/server.py

# Run in development mode with MCP inspector
dev:
	mcp dev src/alertmanager-mcp-server/server.py

# Install in Claude desktop
install:
	mcp install src/alertmanager-mcp-server/server.py

# Docker build
docker-build:
	docker build -t kiennt26/alertmanager-mcp-server:latest .

# Run with Docker
docker-run:
	docker run -p 8000:8000 kiennt26/alertmanager-mcp-server:latest
