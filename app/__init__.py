# Import flask and template operators and import SQLAlchemy
from flask import Flask, render_template
# Import SQLAlchemy
from flask_sqlalchemy import SQLAlchemy
import os

# Define the WSGI application object
app = Flask(__name__)

# Configurations
app.config.from_object(os.environ['APP_SETTINGS'])

# Define the database object which is imported, by modules and controllers
db = SQLAlchemy(app)

# Sample HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

# Build the database:
from app.model.models import Covidinfo

# This will create the database file using SQLAlchemy
db.create_all()
db.session.commit()