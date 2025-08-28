import numpy as np
class Linear_Regression:
    # Initiating the parameters(learning rate and the epochs)
    def __init__(self,learning_rate,epochs):
        self.learning_rate=learning_rate
        self.epochs=epochs


    def fit(self,X,Y):
        # Number of training examples & number of features

        self.m,self.n=X.shape #Number of rows and columns
        # initiating the weight and bias of our model

        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y

        # Implementing gradient descend

        for i in range(self.epochs):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # calculate gradient
        dw = -(2*(self.X.T).dot(self.Y - Y_prediction))/self.m
        db = -2*np.sum(self.Y -Y_prediction)/self.m

        # Updating the weights
        self.w=self.w -self.learning_rate*dw
        self.b=self.b - self.learning_rate*db

    def predict(self,X):
        return X.dot(self.w) + self.b