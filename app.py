from flask import Flask
from flask_cors import CORS
from app.routes import main

def create_app():
    """
    Factory function to create and configure the Flask application.
    """
    app = Flask(__name__)
    
    CORS(app)
    app.config['UPLOAD_FOLDER'] = 'static/uploads' 
    
    app.register_blueprint(main, url_prefix='/')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
