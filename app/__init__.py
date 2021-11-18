from flask import Flask
from config import Config
# from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS, cross_origin
from flask_jwt_extended import JWTManager

app = Flask(__name__)
app.config.from_object(Config)

# Setup the Flask-JWT-Extended extension
app.config["JWT_SECRET_KEY"] = "rfaefqaefasurfaosiefoasefas"
jwt = JWTManager(app)

# bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
# migrate = Migrate(app,db)
CORS(app, support_credentials=True)

from app import routes, models

if __name__ == '__main__':
    app.run()