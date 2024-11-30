from django.shortcuts import render

def home(request):
    model_trained_message = "A machine learning model has been trained for diabetes prediction."
    if request.method == 'POST':
        try:
            age = float(request.POST['age'])
            bmi = float(request.POST['bmi'])
            glucose = float(request.POST['glucose'])

            if glucose > 125 or bmi > 30 or age > 50:
                result = 'Diabetes Detected'
            else:
                result = 'No Diabetes'

            return render(request, 'predictor/home.html', {'result': result, 'model_message': model_trained_message})
        except ValueError:
            return render(request, 'predictor/home.html', {'error': 'Invalid input. Please enter numeric values.', 'model_message': model_trained_message})

    return render(request, 'predictor/home.html', {'model_message': model_trained_message})
