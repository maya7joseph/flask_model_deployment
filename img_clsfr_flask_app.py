from flask import Flask, render_template, request
from image_predictor import get_image_prediction

import os


app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index_v5.html')



@app.route("/", methods = ['POST'])
def prediction():
    
    if request.method == 'GET':
        return render_template('index_v5.html')
    
    if request.method == 'POST':
    
        try:

            print(request.files)

            if 'file' not in request.files:
                print('file not uploaded')
                
                return


            file = request.files['file']
            image = file.read()
            
            
            prediction, confidence = get_image_prediction(image_bytes=image)
                
            return render_template('result.html', prediction = prediction, confidence = confidence)

        except:
            
            return render_template('result.html', prediction = 'Try different image...', confidence = None)
            

          


# if __name__ == '__main__':
# 	port = int(os.environ.get('PORT', 5000))
# 	app.run(host='0.0.0.0', port=port, debug=True)
    
if __name__ == '__main__':
	app.run(debug=True)
