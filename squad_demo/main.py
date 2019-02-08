from flask import Flask, render_template, request
from squad import qa_system_predict


app = Flask("SQuAD")


@app.route("/", methods=["GET", "POST"])
def index():
    context = {}
    if request.method == "POST":
        answer_text, start_word, end_word = qa_system_predict(
            request.form["paragraph"],
            request.form["question"])
        context = dict(
            question=request.form["question"],
            paragraph=request.form["paragraph"],
            answer=answer_text)
    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8881, debug=True)
