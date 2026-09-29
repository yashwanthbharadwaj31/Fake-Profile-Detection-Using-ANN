from django.shortcuts import render
from django.contrib import messages
from django.http import HttpResponse

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# =========================
# PATHS
# =========================
DATASET_PATH = "C:/Users/yashw/OneDrive/Desktop/MIPL-PY-161-USE OF ARTIFICIAL NEURAL NETWORKS TO IDENTIFY FAKE PROFILES/Profile/dataset/dataset.txt"
TEST_PATH = "C:/Users/yashw/OneDrive/Desktop/MIPL-PY-161-USE OF ARTIFICIAL NEURAL NETWORKS TO IDENTIFY FAKE PROFILES/Profile/dataset/test.txt"
MODEL_PATH = "C:/Users/yashw/OneDrive/Desktop/MIPL-PY-161-USE OF ARTIFICIAL NEURAL NETWORKS TO IDENTIFY FAKE PROFILES/Profile/model/ann_model.h5"

# =========================
# BASIC PAGES
# =========================
def index(request):
    return render(request, 'index.html')


def User(request):
    return render(request, 'User.html')


def Admin(request):
    return render(request, 'Admin.html')


def AdminLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username == 'admin' and password == 'admin':
            return render(request, 'AdminScreen.html', {'data': 'Welcome Admin'})
        else:
            return render(request, 'Admin.html', {'data': 'Login Failed'})


# =========================
# DATA HANDLING
# =========================
def importdata():
    data = pd.read_csv(DATASET_PATH)
    data = data.abs()
    return data


def splitdataset(data):
    X = data.values[:, 0:8]
    y = data.values[:, 8].reshape(-1, 1)

    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y)

    train_x, test_x, train_y, test_y = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    return train_x, test_x, train_y, test_y


# =========================
# MODEL TRAINING
# =========================
def GenerateModel(request):
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)

    model = Sequential()
    model.add(Dense(200, input_shape=(8,), activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(lr=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=2)

    loss, accuracy = model.evaluate(test_x, test_y)
    model.save(MODEL_PATH)  # ✅ SAVE MODEL

    acc = accuracy * 100
    return render(request, 'AdminScreen.html', {
        'data': f'ANN Accuracy : {acc:.2f}%'
    })


# =========================
# USER PREDICTION
# =========================
def UserCheck(request):
    if request.method == 'POST':
        data = request.POST.get('t1')

        header = 'Account_Age,Gender,User_Age,Link_Desc,Status_Count,Friend_Count,Location,Location_IP\n'
        content = header + data + "\n"

        with open(TEST_PATH, 'w') as f:
            f.write(content)

        test = pd.read_csv(TEST_PATH)
        test = test.values[:, 0:8]

        # ✅ LOAD TRAINED MODEL
        model = load_model(MODEL_PATH)

        prediction = model.predict(test)
        result = np.argmax(prediction, axis=1)

        if result[0] == 0:
            msg = "Given Account Details Predicted As Genuine"
        else:
            msg = "Given Account Details Predicted As Fake"

        return render(request, 'User.html', {'data': msg})


# =========================
# VIEW TRAINING DATA
# =========================
def ViewTrain(request):
    data = pd.read_csv(DATASET_PATH)

    table = """
    <table border=1 align=center width=100%>
    <tr>
    <th>Account Age</th>
    <th>Gender</th>
    <th>User Age</th>
    <th>Link Description</th>
    <th>Status Count</th>
    <th>Friend Count</th>
    <th>Location</th>
    <th>Location IP</th>
    <th>Profile Status</th>
    </tr>
    """

    for _, row in data.iterrows():
        table += "<tr>"
        for val in row:
            table += f"<td>{val}</td>"
        table += "</tr>"

    table += "</table>"

    return render(request, 'ViewData.html', {'data': table})
