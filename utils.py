#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:38:27 2019

@author: yashvardhan
"""
import os
import pickle

import cv2
import face_recognition
import imutils
from imutils.video import FPS


def create():
    if os.path.exists("encodings1.pickle"):
        print("loading encodings...")
        data = pickle.loads(open("encodings1.pickle", "rb").read())
    else:
        data = {"encodings": [], "names": []}
    print("Checking for new classes...")
    people = os.listdir('dataset')
    for i in people:
        if i not in data['names']:
            for j in os.listdir("dataset/"+i):
                print("processing image {}/{}".format(i, j))
                name = i
                image = cv2.imread('dataset/'+i+'/'+j)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb, model="HOG")
                encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=20)
                for encoding in encodings:
                    data['encodings'].append(encoding)
                    data['names'].append(name)
    print("serializing encodings...")
    f = open("encodings1.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    return data


def new_face():
    print("Enter the name: ")
    loc = 'dataset/'+input()
    try:
        os.makedirs(loc)
        pass
    except FileExistsError:
        print("Directory with same name already exists!")
    pic_no = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            pic_no += 1
            cv2.imwrite(loc+'/'+str(pic_no)+'.jpg', frame)
            if(pic_no > 9):
                cap.release()
                cv2.destroyAllWindows()
                print("Enter more faces?")
                if(input() is 'y'):
                    new_face()
                else:
                    create()
                    return


def recog(data):
    fa = cv2.CascadeClassifier('faces.xml')
    process = 0
    cap = cv2.VideoCapture(0)
    fps = FPS().start()
    while True:
        ret, frame = cap.read()
    # for testing on sample data, remove above two lines and use belwo three lines
    #faces = os.listdir("faces")
    # for i in faces:
        # frame=cv2.imread('faces/'+i)
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=500)
        if (process % 15 is 0):
            #face_locations = face_recognition.face_locations(rgb_frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rects = fa.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
            face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            #matches = face_recognition.compare_faces(data["encodings"], face_encoding,tolerance=0.55)
            name = "No Match"
            mean = {name: 0}
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            matches = [(i < 0.52) for i in face_distances]
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                mean = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                    mean[name] = (mean.get(name, 0)+face_distances[i])
                for i in mean:
                    mean[i] /= counts[i]
                name = min(mean, key=mean.get)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

            cv2.rectangle(frame, (left, bottom+20), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, '{:.2f}'.format(mean[name]), (left + 6, top+15), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, name, (left + 5, bottom+15), font, 0.5, (255, 255, 255), 1)
        process += 1
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        fps.update()
    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Approx. FPS: {:.2f}".format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()
    return mean, face_distances
