from server import app

if __name__ == "__main__":
    app.secret_key = 'ANY_SECRET_KEY'
    app.run(host="0.0.0.0", port=8000,debug=True)
