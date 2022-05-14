from Model import Model
import numpy as np
#exp1

# =========================================================

X = np.array([[1,1],
    [1,-1],
    [-1,1],
    [-1,-1]])

Z = np.array([[1,1],[0,1],[0,1],[0,0]])

S = [2]

funcArray = ["relu", "relu"]

model = Model(X, Z, S, funcArray, 0.2)
for i in range(0, 250):
    y = model.train()
print("relu")
print(y)

# =========================================================

X = np.array([[1,1],
    [1,-1],
    [-1,1],
    [-1,-1]])

Z = np.array([[1,1],[-1,1],[-1,1],[-1,-1]])

S = [2]

funcArray = ["step","step"]

model = Model(X, Z, S, funcArray, 0.6)
for i in range(0, 250):
    y = model.train()
print("step")
print(y)


# ===================================================================


X = np.array([[1,1],
    [1,-1],
    [-1,1],
    [-1,-1]])

Z = np.array([[1,1],[0,1],[0,1],[0,0]])

S = [2]

funcArray = []

model = Model(X, Z, S, funcArray, 0.4)
for i in range(0, 250):
    y = model.train()
print("sigmoid")
print(y)





# ===================================================================



X = np.array([[1,1],
    [1,-1],
    [-1,1],
    [-1,-1]])

Z = np.array([[1,1],[-1,1],[-1,1],[-1,-1]])

S = [2]

funcArray = ["tahn", "tahn"]

model = Model(X, Z, S, funcArray, 0.6)
for i in range(0, 250):
    y = model.train()
print("tahn")
print(y)

# ===================================================================







class CrossValidation():
    # ================ CONSTRUCTOR ========================
    def __init__(self, X, Y, trainPercentage, funcArray = [], learningRate = 0.1, iter = 500, epoch = 250):
        self.X = X
        self.Y = Y
        self.percentage = trainPercentage
        self.indexes = range(X.shape[0])
        self.learningRate  = learningRate
        self.model = model
        self.funcArray = funcArray
        self.iter = iter
        self.epoch = epoch
        
    # ================ FUNCTIONS ========================
    def split(self):
        index = np.sample(self.indexes, int(len(self.indexes)*self.percentage))
        not_index = np.setdiff1d(self.indexes, index)
        x_train = [self.X[i] for i in index]
        x_test = [self.X[i] for i in not_index]
        y_train = [self.Y[i] for i in index]
        y_test = [self.Y[i] for i in not_index]
        return x_train, x_test, y_train, y_test
        
    def accuracy(self, y_pred, y_check):
        res = np.abs(y_check - y_pred)
        return np.sum(res)/len(y_pred)
        
    def test(self):
        accuracies = []
        assertPerc = []
        for i in range(self.iter):
            x_train, x_test, y_train, y_test = self.split()
            for i in range(self.epoch):
                model = Model(x_train, y_train, self.S, self.funcArray, self.learningRate)
                y = model.train()
            y_pred = model.predict(x_test)
            acc = self.accuracy(y_pred, y_test)
            accuracies.append(acc)
            assPerc = self.assertPercentaje(y_pred, y_test)
            assertPerc.append(assPerc)
        return np.mean(assertPerc), np.mean(accuracies)
        
    def assertPercentaje(self, y_pred, y_check):
        res = np.abs(y_check - y_pred)
        res = np.where(res == 0, 1, 0)
        return np.sum(res)/len(y_pred)



X = np.array([[1,1],
    [1,-1],
    [-1,1],
    [-1,-1]])

Z = np.array([[1,1],[0,1],[0,1],[0,0]])

S = [2]

funcArray = []

for i in range(10):
    lr = 1/i
    experimento = CrossValidation(X, Z, 0.8, funcArray, lr, 500)
    
print("sigmoid")
print(S)



# ==============================EXP RANDOMS =======================================

iter = 20
tests = []
for i in range(0, iter):
    layers = i
    S = []
    for j in range(0, layers):
        neurons = np.random.random(10) + 1
        S.append(neurons)
    lr = i*1/iter + 0.01
    funcArray = np.repeat(["step"], layers + 1)
    epoch = np.random.random(500) + 1
    perc = np.random.random(1)
    validation = CrossValidation(X, Z, perc, funcArray, lr, 500, epoch)
    assertP, accuracy = validation.test()
    
    test = {
        "learningRate"  : lr, 
        "splitPercentage"  : perc, 
        "iter"  : 500, 
        "epochs"  : epoch, 
        "accuracy"  : accuracy, 
        "assertPercentage"  : assertP, 
        "layers"  : S, 
        "activationFunctions"  : funcArray}
    tests.append(test)


lrs = []
for i in range(0, len(tests)):
    lrs.append(tests[i]["learningRate"])
plt. 
    
a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
np.savetxt("foo.csv", a, delimiter=",")
    

# ==============================EXP 2 =======================================
# =============================RESULT DATAFRAME======================================

resultColumnNames = ["data_name", "id" , "lr", "splitPercentaje", "iter", "epoch", "accuracy", "assertPerc", "layers", "activationFunctions","meanTime"]

if os.path.isfile("results/result.csv"):
        df = pd.read_csv("results/result.csv")
else:
        df = result

result = pd.DataFrame(rows, columns=resultColumnNames )

        df = df[df.id != name]
        df = pd.concat([df, result], ignore_index = True, axis = 0)

df.to_csv("results/result.csv", index=False, header=True)



dicc: id => struct(lr, splitPercentaje, iter, epoch, accuracy, assertPerc, layers, activationFunctions, meanTime)