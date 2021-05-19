from flask import Flask, render_template, request
import pickle
import numpy as np

model=pickle.load(open('rank.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def home():


    float_features=[float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = int(round(prediction[0], 2))

    return render_template('home.html', prediction_text='Ranking should be  {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)



