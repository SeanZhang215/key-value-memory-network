# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List

from model.memory_network import KVMemoryNetwork
from utils.data_utils import Vocab, multihot

def train_model(
    model: nn.Module,
    train_data: List[Tuple],
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[float], List[float]]:
    """
    Train the Key-Value Memory Network.
    
    Args:
        model: KVMemoryNetwork model
        train_data: List of (questions, keys, values) tuples
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Tuple of training losses and accuracies
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bars
        epoch_pbar = tqdm(range(len(train_data)), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for idx in epoch_pbar:
            questions, keys, values = train_data[idx]
            questions = questions.to(device)
            keys = keys.to(device)
            values = values.to(device)
            
            # Get random examples for comparison
            rand_idx = np.random.randint(0, len(train_data), 2)
            _, rand_keys, rand_values = zip(*[train_data[i] for i in rand_idx])
            
            # Concatenate all keys and values
            all_keys = torch.cat([keys] + list(rand_keys), dim=0).unsqueeze(0)
            all_values = torch.cat([values] + list(rand_values), dim=0).unsqueeze(0)
            
            # Process each question
            for q_idx in range(len(questions)):
                optimizer.zero_grad()
                
                # Forward pass
                question = questions[q_idx].unsqueeze(0)
                output = model(question, all_keys, all_values)
                all_value_embeddings = model.get_value_embeddings(all_values.squeeze(0))
                
                # Compute similarity and loss
                similarity = torch.matmul(output, all_value_embeddings.t())
                target = torch.tensor([q_idx], device=device)
                
                loss = criterion(similarity, target)
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                pred = torch.argmax(similarity).item()
                correct += (pred == q_idx)
                total += 1
                
            # Update progress bar
            avg_loss = epoch_loss / total
            accuracy = 100 * correct / total
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')
    
    return losses, accuracies

def plot_training_curves(losses: List[float], accuracies: List[float]) -> None:
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    vocab_size = 1000
    embed_dim = 128
    model = KVMemoryNetwork(vocab_size, embed_dim)
    
    # Load data and train model
    # train_data = load_data()  # Implementation depends on your data format
    # losses, accuracies = train_model(model, train_data)
    # plot_training_curves(losses, accuracies)