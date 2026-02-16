import os 
import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

root = kagglehub.dataset_download("asjad99/mimiciii")
subdir = os.path.join(root, "mimic-iii-clinical-database-demo-1.4")
lab_path = os.path.join(subdir, "LABVENTS.csv")
df = pd.read_csv(lab_path).head(50000)

if "FLAG" in df.columns:
    df["FLAG_CODE"] = pd.factorize(df["FLAG"])[0]
if "VALUEUOM" in df.columns:
    df["VALUEUOM_CODE"] = pd.factorize(df["VALUEUOM"])[0]
numeric_cols = []
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_cols.append(c)
for extra in ["FLAG_CODE", "VALUEUOM_CODE"]:
    if extra in df.columns and extra not in numeric_cols:
        numeric_cols.append(extra)
feature_cols = numeric_cols[:10]

if "FLAG_CODE" in df.columns:
    df2 = df[feature_cols + ["FLAG_CODE"]].dropna().copy()
    med = df2["FLAG_CODE"].median()
    y_np = (df2["FLAG_CODE"].values > med).astype(int)
else:
    df2 = df[feature_cols].dropna().copy()
    med = df2[feature_cols[0]].median()
    y_np = (df2[feature_cols[0]].values > med).astype(int)

X= torch.tensor(df2[feature_cols].values, dtype=torch.float32)
y= torch.tensor(y_np, dtype=torch.long)

if X.shape[1] < 10:
    reps = (10 + X.shape[1] - 1) // X.shape[1]
    X = X.repeat(1, reps)[:, :10]
elif X.shape[1] > 10:
    X = X[:, :10]

print("X shape:", X.shape, "y shape:", y.shape)

#Spitting into forget and retain sets
N_forget = 2000
X_forget, y_forget = X[:N_forget], y[:N_forget]
X_retain, y_retain = X[N_forget:], y[N_forget:]
#Creating dataloaders
forget_loader = DataLoader(TensorDataset(X_forget, y_forget), batch_size=128, shuffle=True)
retain_loader = DataLoader(TensorDataset(X_retain, y_retain), batch_size=128, shuffle=True)

#Model 
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

#Train the original model on all the data 
train_loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
print(f"Train epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f }")

#Accuracy on the loader
def accuracy_on_loader (m,loader):
    m.eval()
    correct, total = 0,0
    with torch.no_grad():
        for xb, yb in loader:
            pred = m(xb).argmax(dim=1)
            correct += int((pred == yb).sum())
            total += yb.numel()
    return 100.0 * correct / max(total, 1)

acc_forget_before = accuracy_on_loader(model, forget_loader)
acc_retain_before = accuracy_on_loader(model, retain_loader)
print(f"Before unlearning - Forget set acc: {acc_forget_before:.2f}%, Retain set acc: {acc_retain_before:.2f}%")

#Finding PSG Unlearning using fixed indexing
class PerturbedSignGradientUnlearning:
    def __init__(self, model, noise_scale=0.1):
        self.model = model
        self.noise_scale = noise_scale
    def unlearn_step(self, X_forget, y_forget, learning_rate=0.01, lambda_retain=1.0):
        self.model.zero_grad()
        out_f = self.model(X_forget)
        loss_forget = nn.CrossEntropyLoss()(out_f, y_forget)
        loss_forget.backward()
        perturbed_grads= [] 
        for p in self.model.parameters():
            if p.grad is None:
               perturbed_grads.append(None)
               continue 
            sg = torch.sign(p.grad)
            noise = torch.randn_like(sg) * self.noise_scale
            perturbed_grads.append((sg + noise).detach())
        self.model.zero_grad()
        out_retain = self.model(X_retain)
        loss_retain = nn.CrossEntropyLoss()(out_retain, y_retain)
        loss_retain.backward()
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if p.grad is None or perturbed_grads[i] is None:
                    continue
                update = (-perturbed_grads[i] + lambda_retain * p.grad)
                p.data -= learning_rate * update
        return float(loss_forget), float(loss_retain)
    
    def unlearn(self,forget_loader, retain_loader, epochs=10, learning_rate=0.01, lambda_retain=1.0):
        for epoch in range(epochs):
            total_forget_loss = 0.0
            total_retain_loss = 0.0
            n = 0.0
            for (Xf, yf), (Xr, yr) in zip(forget_loader, retain_loader):
                fl, rl = self.unlearn_step(Xf, yf, learning_rate, lambda_retain)
                total_forget_loss += fl
                total_retain_loss += rl
                n += 1.0
            print(f"Epoch {epoch+1}/{epochs}: Forget Loss = {total_forget_loss/n:.4f} Retain Loss = {total_retain_loss/n:.4f}")


model.train()
unlearner = PerturbedSignGradientUnlearning(model, noise_scale=0.1)
unlearner.unlearn(forget_loader, retain_loader, epochs=10, learning_rate=0.01, lambda_retain=1.0)

acc_forget_after = accuracy_on_loader(model, forget_loader)
acc_retain_after = accuracy_on_loader(model, retain_loader)
print(f"After unlearning - Forget set acc: {acc_forget_after:.2f}%, Retain set acc: {acc_retain_after:.2f}%")

