
# Import the required libraries
from flask import Flask, render_template 

# Define the API server
app = Flask(__name__)

@app.route('/')
def landing_page():
    return render_template("home.html")

# Define the API endpoint for object detection
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    print("request received")

if __name__ == '__main__':
    app.run(debug=True)

