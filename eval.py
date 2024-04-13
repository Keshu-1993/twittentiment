from sklearn.metrics import accuracy_score, classification_report
import torch
from model import model, criterion, optimizer
#from data_prep import train_data, X_train, y_train 
from feat_engg import X_validation, y_validation, X_train, y_train
import os

# Define the path to save the model
model_dir = '/Users/keshav/Documents/For_Interviews/ClLo_tech/saved_model/'

# # Define the base filename for the model
base_filename = 'trained_model.pth'

# List all files in the model directory
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

# Sort the model files based on modification time
model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)


def evaluate_model(model, X, y):

    # Set model to evaluation mode
    model.eval()
    
    # Convert data to PyTorch tensors
    X_tensor = torch.Tensor(X.toarray())
    y_tensor = torch.Tensor(y.values).long()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_tensor, predicted)
    print('Accuracy:', accuracy)
    
    # Print classification report
    print(classification_report(y_tensor, predicted))

# Train model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=10):

    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        
        # Convert data to PyTorch tensors
        X_tensor = torch.Tensor(X_train.toarray())
        y_tensor = torch.Tensor(y_train.values).long()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_tensor)
        print('Forward pass:', outputs.size())
        loss = criterion(outputs, y_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# Train the model
train_model(model, criterion, optimizer, X_train, y_train)

# Evaluate the model
evaluate_model(model, X_validation, y_validation)

# Save the trained model
#torch.save(model, '/Users/keshav/Documents/For_Interviews/ClLo_tech/saved_model/trained_model.pth')

# # Check if the file already exists
# if os.path.exists(os.path.join(model_dir, base_filename)):
#     # If the file exists, increment the counter until finding a filename that doesn't exist
#     i = 1
#     while os.path.exists(os.path.join(model_dir, f'{base_filename.split(".")[0]}_{i}.pth')):
#         i += 1
#     # Save the model with the new filename
#     torch.save(model, os.path.join(model_dir, f'{base_filename.split(".")[0]}_{i}.pth'))
# else:
#     # If the file doesn't exist, save the model with the base filename
#     torch.save(model, os.path.join(model_dir, base_filename))

# Load the previously trained model
# trained_model_path = os.path.join(model_dir, base_filename)
# if os.path.exists(trained_model_path):
#     model = torch.load(trained_model_path)
# else:
#     print(f"No model found at {trained_model_path}. Please train a new model first.")

#Check if there are any model files
if model_files:
    # Load the latest trained model
    latest_model_path = os.path.join(model_dir, model_files[0])
    model = torch.load(latest_model_path)
    print(f"Loaded the latest trained model from: {latest_model_path}")
else:
    print("No model found in the model directory. Please train a new model first.")

# Retrain the model
train_model(model, criterion, optimizer, X_train, y_train)

# Evaluate the retrained model
evaluate_model(model, X_validation, y_validation)

# Save the retrained model
i = 1
while os.path.exists(os.path.join(model_dir, f'{base_filename.split(".")[0]}_{i}.pth')):
    i += 1
# Save the model with the new filename
torch.save(model, os.path.join(model_dir, f'{base_filename.split(".")[0]}_{i}.pth'))