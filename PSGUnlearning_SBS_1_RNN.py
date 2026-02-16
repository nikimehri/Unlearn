import os
import kagglehub
import pandas as pd 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

root = kagglehub.dataset_download("asjad99/mimiciii")
subdir = os.path.join(root, "mimic-iii-clinical-database-demo-1.4")
lab_path = os.path.join(subdir, "LABEVENTS.csv")
df= pd.read_csv(lab_path)
df=df.head(50000)
numeric_cols = []
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_cols.append(c)

if "FLAG" in df.columns:
    df["FLAG_CODE"] = pd.factorize(df["FLAG"])[0]
    numeric_cols.append("FLAG_CODE")
if "VALUEUOM" in df.columns:
    df["VALUEUOM_CODE"] = pd.factorize(df["VALUEUOM"])[0]
    numeric_cols.append("VALUEUOM_CODE")

feature_cols = numeric_cols [:10]
if "FLAG_CODE" in df.columns:
    df2 = df[feature_cols + ["FLAG_CODE"]].dropna().copy()
    median_val = df2["FLAG_CODE"].median()
    y=(df2["FLAG_CODE"].values > median_val).astype(int)
else:
    df2= df[feature_cols].dropna().copy()
    median_val = df2[feature_cols[0]].median()
    y=(df2[feature_cols[0]].values > median_val).astype(int)


#Creating Torch Tensors 
X= torch.tensor(df2[feature_cols].values, dtype=torch.float32)
y= torch.tensor(y, dtype=torch.long)

if X.shape[1] < 10:
    reps = (10 + X.shape[1] - 1) // X.shape[1]
    X = X.repeat(1, reps)[:, :10]
elif X.shape[1] > 10:
    X = X[:, :10]

#Split into forget set 
X_forget = X[:32]
y_forget = y[:32]

#PSG demo models 
class RNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


model = RNNClassifier(input_size=1, hidden_size=64, num_layers=1, num_classes=2)

outputs = model(X_forget)
loss_forget = nn.CrossEntropyLoss()(outputs, y_forget) #What does cross entropy do?
loss_forget.backward ()
grad= model.rnn.weight_ih_l0.grad
print (f"Standard Gradient:\n{grad[0,:5]}")
sign_grad = torch.sign(grad)
print (f"Sign Gradient:\n{sign_grad[0,:5]}")
noise = torch.randn_like(sign_grad) * 0.1 #why is sign_grad used here? 
pertubated_sign_grad = sign_grad + noise 
print (f"Pertubated Sign Gradient:\n{pertubated_sign_grad[0,:5]}")

learning_rate = 0.01 #How does changing the learning rate affect the unlearning process?
with torch.no_grad():
    for param in model.parameters():
        if param.grad is not None:
            sg = torch.sign(param.grad)
            eps = torch.randn_like(sg) * 0.1
            psg=sg + eps
            param += learning_rate * psg

print ("PSG Update Done.")

#Graphs 
N=200
g_std= grad.detach().flatten()[:N].cpu()
g_sign = sign_grad.detach().flatten()[:N].cpu()
g_psg = pertubated_sign_grad.detach().flatten()[:N].cpu()

#Graphing gradient 
plt.figure()
plt.plot(g_std.numpy())
plt.title("Standard Gradient (first N params)")
plt.xlabel("Parameter index")
plt.ylabel("Gradient value")
plt.axhline(0)
plt.savefig("grad_standard.png", dpi=200)

#Graphing sign gradient
plt.figure()
plt.plot(g_sign.numpy())
plt.title("Sign Gradient (first N params)")
plt.xlabel("Parameter index")
plt.ylabel("Sign(grad)")
plt.axhline(0)
plt.savefig("grad_sign.png", dpi=200)

#Graphibg pertubated sign gradient
plt.figure()
plt.plot(g_psg.numpy())
plt.title("Perturbed Sign Gradient (first N params)")
plt.xlabel("Parameter index")
plt.ylabel("Sign(grad) + noise")
plt.axhline(0)
plt.savefig("grad_perturbed.png", dpi=200)


with torch.no_grad():
    outputs_after = model(X_forget)
    loss_after = nn.CrossEntropyLoss()(outputs_after, y_forget)
print("Forget loss before:", float(loss_forget), "after:", float(loss_after))
