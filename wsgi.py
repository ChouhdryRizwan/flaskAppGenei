from server import app

if __name__ == "__main__":
    app.secret_key = 'ANY_SECRET_KEY'
    app.run(debug=True)
