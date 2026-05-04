from flask import Flask
from flask_cors import CORS

from controllers.svm_controller import svm_blueprint
from controllers.dl_controller import dl_blueprint
from controllers.cnn_controller import cnn_blueprint
from controllers.clustering_controller import clustering_blueprint
from controllers.fuzzy_controller import fuzzy_blueprint
from controllers.som_controller import som_blueprint

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.register_blueprint(svm_blueprint)
app.register_blueprint(dl_blueprint)
app.register_blueprint(cnn_blueprint)
app.register_blueprint(clustering_blueprint)
app.register_blueprint(fuzzy_blueprint)
app.register_blueprint(som_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
