from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import and register routes from main.py
    from app.main import register_routes
    register_routes(app)

    return app