# Use an official Python 3.9 slim image as the base image
FROM python:3.9-slim

# Set environment variables to disable bytecode generation and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building Cartopy and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libproj-dev \
    proj-data \
    proj-bin \
    gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install dependencies from the requirements file
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install pip-audit

# Run a security audit before launching the app
RUN pip-audit -r requirements.txt || true  

# Copy the entire project code into the container
COPY . .

# Expose port 8001 for the FastAPI app
EXPOSE 8001

# Command to run the FastAPI app using uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]