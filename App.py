from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        msg = request.form["message"]
        msg_tfidf = vectorizer.transform([msg])
        prediction = model.predict(msg_tfidf)[0]

        if prediction == "spam":
            result = "SPAM"
        else:
            result = "NOT SPAM"
    return render_template("spam_detector.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)


