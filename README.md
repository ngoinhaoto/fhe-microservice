# FHE Microservice

This microservice handles privacy-preserving face embedding operations using Homomorphic Encryption (CKKS via TenSEAL). It provides RESTful APIs for encrypted face registration, verification, and similarity computation, and integrates with the main application server.

## Features

- FastAPI-based REST API
- TenSEAL CKKS context generation and management
- Endpoints for encrypted face registration and verification
- Secure context separation (public/secret)
- Integration with main server and React client

## Installation

### 1. Clone the repository

```sh
git clone https://github.com/your-org/tenseal-system.git
cd tenseal-system/fhe-microservice
```

### 2. Create and activate a Python environment

```sh
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and edit as needed:

```sh
cp .env.example .env
```

### 5. Run the microservice

```sh
python main.py
```

The API will be available at [http://localhost:8002](http://localhost:8002).

## Environment Variables

See `.env.example` for required variables.

## Usage

- Integrate with the main FastAPI server and React client.
- Use endpoints for encrypted face registration and verification.
- Context files are managed in the `context/` directory.

## Troubleshooting

- Ensure the context directory exists and is writable.
- If context files are missing, they will be generated automatically on first run.
- Check `.env` for correct configuration.
