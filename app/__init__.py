from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Allow requests from any origin
    app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory for uploaded files

    from .routes import main
    app.register_blueprint(main)

    return app
