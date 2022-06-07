import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json= { '10th %':89.5, '12th %':87.8, 'CGPA':8.7,'Backlogs':0,'coding':1,'stay':1,'physical_activites':1,'communication':3,'no_of_online_courses':3,'Attendance':2})

print(r.json())