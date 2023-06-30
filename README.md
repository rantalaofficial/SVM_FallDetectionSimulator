# SVM_FallDetectionSimulator
A simple fall detection simulator made with pygame graphics, pymunk physics and sklearn AI library.

The simulator detects falls by Support Vector Machine (SVM) -AI model. The AI is trained by first gathering data with main.py, by jumping and falling around. The "human" is controller by WASD. After that, data is dumped to data.txt and train.py can be ran. It generates a SVM_MODEL.joblib file that has the model saved. Now when you open the simulator main.py once again, you should see predictions about the falling.

Depencies:
numpy
sklearn
joblib
pymunk
pygame

Screenshot:
![image](https://github.com/rant4la/SVM_FallDetectionSimulator/assets/33716618/6c95dd18-7203-4a32-a01e-fc4c2cd5b8d3)
