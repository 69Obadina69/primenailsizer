"""
Main backend entrypoint.

NOTE ON FRAMEWORK: the spec calls for FastAPI, but this sandbox has no
network access and FastAPI/uvicorn/pydantic are not preinstalled, so they
could not be installed here. Flask IS preinstalled and was used instead so
the pipeline could actually be built and run end-to-end in this
environment. The code is organized so porting to FastAPI later is a
routes-only change (`pip install fastapi uvicorn pydantic`, then swap
`api/routes_*.py` Flask blueprints for FastAPI routers) — none of the
vision/ business logic needs to change.
"""
from flask import Flask
from api.routes_health import health_bp
from api.routes_reference import reference_bp
from api.routes_measure import measure_bp


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB upload limit

    app.register_blueprint(health_bp)
    app.register_blueprint(reference_bp)
    app.register_blueprint(measure_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)
