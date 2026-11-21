# DVC and MLFlow Lab Step 3 - Digit Classifier API

This is a simple REST API for handwritten digit classification using a **Scikit-learn Random Forest Classifier**, as setup
in [MLFlowSetup.ipynb](MLFlowSetup.ipynb).
The model is served through FastAPI and packaged in a Docker container.

## Features

- **Machine Learning:** Random Forest Classifier trained on Scikit-learn's handwritten digits dataset.
- **API:** FastAPI endpoint for digit prediction.
- **Model Tracking:** Model trained and tracked using MLflow on DagsHub.
- **Containerization:** Docker image containing the API and trained model.
- **CI/CD:** GitHub Actions workflow to automatically build and publish the Docker image to Docker Hub.

## Tech Stack

- Python
- Scikit-learn
- MLflow
- FastAPI & Uvicorn
- Docker
- GitHub Actions
- DagsHub

## API Usage

The API exposes a single endpoint:

`POST /predict`

It accepts a JSON array containing exactly **64 integers**, representing an 8×8 grayscale image flattened row by row. 
Pixel values must range from 0 to 16.

**Example request:**

```json
[
  0, 0, 12, 10, 0, 0, 0, 0,
  0, 0, 14, 16, 16, 14, 0, 0,
  0, 0, 13, 16, 15, 10, 1, 0,
  0, 0, 11, 16, 16, 7, 0, 0,
  0, 0, 0, 4, 7, 16, 7, 0,
  0, 0, 0, 0, 4, 16, 9, 0,
  0, 0, 5, 4, 12, 16, 4, 0,
  0, 0, 9, 16, 16, 10, 0, 0
]
```

**Example response:**

```json
{
  "prediction": 5
}
```

Interactive API documentation is available at `/docs`.

## Running with Docker

Pull the image:

```bash
docker pull elo0/digit-classifier:latest
```

Run the container:

```bash
docker run -d --name digit-classifier -p 8000:8000 elo0/digit-classifier:latest
```

The API is then available at:

`http://localhost:8000`

Interactive documentation:

`http://localhost:8000/docs`

## CI/CD Pipeline

A GitHub Actions workflow automatically builds and publishes the Docker image when changes are pushed to the `main` 
branch.

During the Docker build, the trained model is retrieved from DagsHub using MLflow. Authentication credentials are 
supplied through GitHub Secrets.

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   └── main.py
├── .dvc/
│   ├── config
│   └── .gitignore
├── data/
│   └── raw_dataset.csv.dvc  # Not used in API
├── .github/
│   └── workflows/
│       └── docker-build.yml
├── Dockerfile
├── requirements.txt
├── MLFlowSetup.ipynb
├── .dockerignore
├── .dvcignore
├── .gitignore
└── README.md
```

## Author

Elodie DUFLAUT, 2026