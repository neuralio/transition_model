# Backend API for TRANSITION model 

This repository contains a FastAPI application for the TRANSITION model 

The application is dockerized using Docker Compose and uses Python logging to record events and errors to a file named `app.log`. By mounting the container’s `/app` directory as a volume, the log file is persisted on the host machine for easy access.




---

## Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## Building and Running with Docker Compose

### 1. Build the Docker Image

From the project root, run:

```bash
docker compose build


This command builds the Docker image using the provided Dockerfile.

2. Run the Application

Start the containers by running:
docker compose up

The FastAPI application will start and listen on port 8001. It will be accessible at https://transitionapi.neuralio.ai/docs

Logging
The application uses Python’s logging module with a rotating file handler that writes log entries to app.log inside the container. With the Docker Compose volume configuration, this file is also available on your host.

Viewing the Log File
You can view the log file on your host machine with:
cat app.log
Or, to follow live updates:

tail -f app.log
Testing the API with curl
Once the application is running, you can test the API endpoints using curl:

PV Past Endpoint
curl -X POST https://transitionapi.neuralio.ai/pv_past -H 'Content-Type: application/json' -d '{}'

PV Future Endpoint
curl -X POST https://transitionapi.neuralio.ai/pv_future -H 'Content-Type: application/json' -d '{}'

Crop Past Endpoint
curl -X POST https://transitionapi.neuralio.ai/crop_past -H 'Content-Type: application/json' -d '{}'

Crop Future Endpoint
curl -X POST https://transitionapi.neuralio.ai/crop_future -H 'Content-Type: application/json' -d '{}'

Base RL Past Endpoint 
curl -X POST https://transitionapi.neuralio.ai/base_past -H 'Content-Type: application/json' -d '{}'

Base RL Future Endpoint 
curl -X POST https://transitionapi.neuralio.ai/base_future -H 'Content-Type: application/json' -d '{}'

PECS RL Past Endpoint 
curl -X POST https://transitionapi.neuralio.ai/pecs_past -H 'Content-Type: application/json' -d '{}'

PECS RL Future Endpoint 
curl -X POST https://transitionapi.neuralio.ai/pecs_future -H 'Content-Type: application/json' -d '{}'

FULL 1 RL Past Endpoint 
curl -X POST https://transitionapi.neuralio.ai/full1_past -H 'Content-Type: application/json' -d '{}'

FULL 1 RL Future Endpoint 
curl -X POST https://transitionapi.neuralio.ai/full1_future -H 'Content-Type: application/json' -d '{}'

FULL 2 RL Past Endpoint 
curl -X POST https://transitionapi.neuralio.ai/full2_past -H 'Content-Type: application/json' -d '{}'

FULL 2 RL Future Endpoint 
curl -X POST https://transitionapi.neuralio.ai/full2_future -H 'Content-Type: application/json' -d '{}'


