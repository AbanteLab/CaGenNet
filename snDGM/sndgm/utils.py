def load_data(file_path):
    # Function to load data from a specified file path
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Function to preprocess the data
    # Example: Normalization or scaling can be added here
    return (data - data.mean()) / data.std()

def split_data(data, test_size=0.2, random_state=42):
    # Function to split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def save_model(model, file_path):
    # Function to save the trained model
    torch.save(model.state_dict(), file_path)

def load_model(model, file_path):
    # Function to load a trained model
    model.load_state_dict(torch.load(file_path))
    return model