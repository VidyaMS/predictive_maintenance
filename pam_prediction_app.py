from flask import Flask, render_template, request, send_from_directory
import pickle as p
import numpy as np
import pandas as pd
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'pam_uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def features():
   return render_template('Upload_file_for_pam.html')

## number of uploads
count = 0
 

@app.route('/predict' , methods =  ['GET' , 'POST'])
def upload_and_predict():
	global count
	if request.method == 'POST':
		## read in the file  for model prediction.
		f = request.files['file']
		##
		file = f.filename
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], file))
		test = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], file))
		cols = test.columns.tolist()
		
		## check for null values and drop them
		test.dropna(inplace =True)
		
		## check for rows not equal to zero
		if test.shape[0] == 0:
			return render_template("file_error.html")
		## scale the input numeric values
		X_test = test[cols[3:8]]
		##
		X_test_scaled = scaler.transform(X_test)
		y_test = test['Machine failure']
                
		## predict for the uploaded file
		predicted_class = model.predict(X_test_scaled)
		test['predicted_class'] = predicted_class
		test['probability'] = model.predict_proba(X_test_scaled)[:,1]
		(tn,fp,fn,tp) = confusion_matrix(y_test, predicted_class).ravel()
		result_dict =  {'Correctly predicted failures ': tp ,'Falsely predicted failures': fn, 'Correctly predicted non-failures': tn , 'Falsely predicted non-failures' : fp}

                ## save the file in the download folder 
		count = count + 1
		file_path = os.getcwd()+'/pam_results/'+'predictions' + '_' + str(count) + '.csv'
		test.to_csv(file_path, index =False)

		return render_template("machine_failure_prediction_result.html" , result = result_dict)

@app.route('/download' , methods = ['GET', 'POST'])
def download_file():
	global count
	try:
		#return send_file('predictions.csv')
		file_path = os.getcwd()+'/pam_results/'
		filename =  'predictions' + '_' + str(count) + '.csv'
		return send_from_directory(file_path,filename, as_attachment =True , cache_timeout = 0)
	except Exception as e:
		return str(e)


if __name__ == '__main__':
	modelfile = 'pm_ai2020_model_2.sav'
	model = p.load(open(modelfile, 'rb'))
	scaler = p.load(open('minmax_scaler.sav','rb'))
	print("model and scaler loaded")
	app.run(debug = True,host = '0.0.0.0' , port = 5000 )

