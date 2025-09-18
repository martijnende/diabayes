from flask import Blueprint, jsonify, render_template, request

# Create a Blueprint so routes can be registered to the app in __init__.py
bp = Blueprint("main", __name__)

@bp.route("/")
def index():
    return render_template("index.html")  # you can put templates/ under gui/

@bp.route("/api/ping")
def ping():
    return jsonify({"status": "ok", "message": "pong"})

@bp.route("/api/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify({"you_sent": data})

@bp.route("/hello/<name>")
def hello(name):
    return f"Hello, {name}!"

